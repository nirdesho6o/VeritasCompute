"""Phase 1d tests for the mechanism engine."""

from core.mechanism import solve_knapsack_vcg, settle_brier, update_calibration_weights
from core.schemas import VerifierBid


def _bid(vid: str, p: float, voi: float, cost: int, script: str = "pass") -> VerifierBid:
    return VerifierBid(
        verifier_id=vid,
        prob_violation=p,
        expected_voi=voi,
        compute_cost=cost,
        script=script,
    )


class TestKnapsackOptimalWelfare:
    """Knapsack selects optimal welfare combination under budget."""

    def test_picks_highest_value_within_budget(self):
        bids = [
            _bid("v1", 0.5, 10.0, 3),
            _bid("v2", 0.5, 8.0, 2),
            _bid("v3", 0.5, 7.0, 2),
        ]
        weights = {"v1": 1.0, "v2": 1.0, "v3": 1.0}
        bankrolls = {"v1": 100.0, "v2": 100.0, "v3": 100.0}
        result = solve_knapsack_vcg(bids, max_compute=4, calibration_weights=weights, bankrolls=bankrolls)
        # v2(8,cost2) + v3(7,cost2) = 15 welfare in cost 4 beats v1(10,cost3) + nothing or v1+v2(cost5>4)
        assert set(result.winners) == {"v2", "v3"}
        assert result.total_welfare == 15.0

    def test_single_winner_when_budget_tight(self):
        bids = [
            _bid("v1", 0.5, 20.0, 5),
            _bid("v2", 0.5, 12.0, 3),
            _bid("v3", 0.5, 10.0, 3),
        ]
        weights = {"v1": 1.0, "v2": 1.0, "v3": 1.0}
        bankrolls = {"v1": 100.0, "v2": 100.0, "v3": 100.0}
        result = solve_knapsack_vcg(bids, max_compute=5, calibration_weights=weights, bankrolls=bankrolls)
        # v2+v3 = 22 in cost 6 > budget. v1 = 20 in cost 5 fits. v2+nothing=12. So v1 wins.
        assert result.winners == ["v1"]
        assert result.total_welfare == 20.0

    def test_effective_bid_capped_by_bankroll(self):
        bids = [
            _bid("v1", 0.5, 50.0, 2),
            _bid("v2", 0.5, 8.0, 2),
        ]
        weights = {"v1": 1.0, "v2": 1.0}
        bankrolls = {"v1": 5.0, "v2": 100.0}  # v1 bankroll caps effective to 5
        result = solve_knapsack_vcg(bids, max_compute=2, calibration_weights=weights, bankrolls=bankrolls)
        assert result.winners == ["v2"]  # v2 effective=8 > v1 effective=5

    def test_calibration_weight_scales_effective_bid(self):
        bids = [
            _bid("v1", 0.5, 20.0, 2),
            _bid("v2", 0.5, 8.0, 2),
        ]
        weights = {"v1": 0.3, "v2": 1.0}  # v1 effective = 0.3*20 = 6 < v2's 8
        bankrolls = {"v1": 100.0, "v2": 100.0}
        result = solve_knapsack_vcg(bids, max_compute=2, calibration_weights=weights, bankrolls=bankrolls)
        assert result.winners == ["v2"]


class TestClarkePivotPricing:
    """Clarke Pivot price >= 0 and <= effective bid for every winner."""

    def test_prices_non_negative_and_bounded(self):
        bids = [
            _bid("v1", 0.5, 10.0, 2),
            _bid("v2", 0.5, 8.0, 2),
            _bid("v3", 0.5, 7.0, 2),
        ]
        weights = {"v1": 1.0, "v2": 1.0, "v3": 1.0}
        bankrolls = {"v1": 100.0, "v2": 100.0, "v3": 100.0}
        result = solve_knapsack_vcg(bids, max_compute=4, calibration_weights=weights, bankrolls=bankrolls)
        effective = {"v1": 10.0, "v2": 8.0, "v3": 7.0}
        for vid in result.winners:
            price = result.clarke_prices[vid]
            assert price >= 0, f"Clarke price for {vid} is negative: {price}"
            assert price <= effective[vid], f"Clarke price {price} > effective bid {effective[vid]} for {vid}"

    def test_prices_with_varied_costs(self):
        bids = [
            _bid("v1", 0.5, 15.0, 3),
            _bid("v2", 0.5, 12.0, 2),
            _bid("v3", 0.5, 5.0, 1),
        ]
        weights = {v: 1.0 for v in ["v1", "v2", "v3"]}
        bankrolls = {v: 100.0 for v in ["v1", "v2", "v3"]}
        result = solve_knapsack_vcg(bids, max_compute=5, calibration_weights=weights, bankrolls=bankrolls)
        for vid in result.winners:
            price = result.clarke_prices[vid]
            eff = min(bankrolls[vid], weights[vid] * {b.verifier_id: b.expected_voi for b in bids}[vid])
            assert price >= 0
            assert price <= eff


class TestSybilBankruptcy:
    """A Sybil critic (always p=0.99, voi=50 on benign y=0 tasks) goes bankrupt fast."""

    def test_sybil_depletes_within_6_rounds(self):
        starting_bankroll = 100.0
        bankrolls = {"sybil": starting_bankroll, "honest": starting_bankroll}
        loss_histories: dict[str, list[float]] = {"sybil": [], "honest": []}
        weights = {"sybil": 1.0, "honest": 1.0}
        alpha = 12.0

        for _ in range(10):
            # All rounds are benign (y=0)
            outcome = 0
            sybil_bid = _bid("sybil", 0.99, 50.0, 2)
            honest_bid = _bid("honest", 0.1, 30.0, 2)
            bids = [sybil_bid, honest_bid]

            auction = solve_knapsack_vcg(bids, max_compute=4, calibration_weights=weights, bankrolls=bankrolls)
            settlements = settle_brier(bids, outcome, auction, bankrolls, alpha=alpha)

            for s in settlements:
                bankrolls[s.verifier_id] = s.new_bankroll
                loss_histories[s.verifier_id].append(s.brier_loss)

            weights = update_calibration_weights(loss_histories)

        assert bankrolls["sybil"] < starting_bankroll / 4, (
            f"Sybil bankroll {bankrolls['sybil']} should be < {starting_bankroll / 4}"
        )
        assert weights["sybil"] < 0.25, (
            f"Sybil calibration weight {weights['sybil']} should be < 0.25"
        )

    def test_honest_critic_accumulates(self):
        """An honest critic (p~0.1 on benign, p~0.9 on violations) grows bankroll."""
        starting_bankroll = 100.0
        bankrolls = {"honest": starting_bankroll}
        loss_histories: dict[str, list[float]] = {"honest": []}
        weights = {"honest": 1.0}
        alpha = 10.0

        tasks = [0, 0, 0, 1, 1, 0, 1, 0, 0, 1]  # mix of benign and violation
        for outcome in tasks:
            p = 0.1 if outcome == 0 else 0.9
            bid = _bid("honest", p, 20.0, 2)
            bids = [bid]

            auction = solve_knapsack_vcg(bids, max_compute=4, calibration_weights=weights, bankrolls=bankrolls)
            settlements = settle_brier(bids, outcome, auction, bankrolls, alpha=alpha)

            for s in settlements:
                bankrolls[s.verifier_id] = s.new_bankroll
                loss_histories[s.verifier_id].append(s.brier_loss)

            weights = update_calibration_weights(loss_histories)

        assert bankrolls["honest"] > starting_bankroll, (
            f"Honest bankroll {bankrolls['honest']} should exceed starting {starting_bankroll}"
        )
