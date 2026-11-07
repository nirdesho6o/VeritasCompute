"""VeritasMechanismEngine: 0-1 Knapsack VCG + Centered Brier Strictly Proper Scoring Rule."""

from __future__ import annotations

from core.schemas import AuctionResult, BrierSettlement, VerifierBid


def _compute_effective_bids(
    bids: list[VerifierBid],
    calibration_weights: dict[str, float],
    bankrolls: dict[str, float],
) -> dict[str, float]:
    """v_tilde_i = min(W_i, w_i * v_i)"""
    effective = {}
    for b in bids:
        w_i = calibration_weights.get(b.verifier_id, 1.0)
        effective[b.verifier_id] = min(
            bankrolls.get(b.verifier_id, 0.0),
            w_i * b.expected_voi,
        )
    return effective


def _solve_01_knapsack(
    items: list[tuple[str, float, int]],
    capacity: int,
) -> list[str]:
    """Solve 0-1 knapsack via DP. items = [(id, value, cost)]. Returns winner ids."""
    n = len(items)
    dp = [[0.0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        _, val, cost = items[i - 1]
        for c in range(capacity + 1):
            dp[i][c] = dp[i - 1][c]
            if cost <= c and dp[i - 1][c - cost] + val > dp[i][c]:
                dp[i][c] = dp[i - 1][c - cost] + val

    winners = []
    c = capacity
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            winners.append(items[i - 1][0])
            c -= items[i - 1][2]

    return winners


def solve_knapsack_vcg(
    bids: list[VerifierBid],
    max_compute: int,
    calibration_weights: dict[str, float],
    bankrolls: dict[str, float],
) -> AuctionResult:
    effective = _compute_effective_bids(bids, calibration_weights, bankrolls)
    bid_map = {b.verifier_id: b for b in bids}
    items = [
        (b.verifier_id, effective[b.verifier_id], b.compute_cost)
        for b in bids
    ]

    winners = _solve_01_knapsack(items, max_compute)
    winner_set = set(winners)
    allocations = {b.verifier_id: b.verifier_id in winner_set for b in bids}
    total_welfare = sum(effective[vid] for vid in winners)

    # Clarke Pivot pricing: q_i = optimal_welfare_without_i - welfare_of_others_in_current
    clarke_prices: dict[str, float] = {}
    for vid in winners:
        others_in_current = sum(effective[w] for w in winners if w != vid)
        items_without_i = [it for it in items if it[0] != vid]
        winners_without_i = _solve_01_knapsack(items_without_i, max_compute)
        optimal_without_i = sum(effective[w] for w in winners_without_i)
        clarke_prices[vid] = optimal_without_i - others_in_current

    return AuctionResult(
        winners=winners,
        allocations=allocations,
        clarke_prices=clarke_prices,
        total_welfare=total_welfare,
    )


def settle_brier(
    bids: list[VerifierBid],
    outcome: int,
    auction_result: AuctionResult,
    bankrolls: dict[str, float],
    alpha: float = 10.0,
) -> list[BrierSettlement]:
    """Centered Brier SPSR settlement. ΔW = α * ((1 - (y-p)²) - 0.75) - x*·q"""
    winner_set = set(auction_result.winners)
    settlements = []
    for b in bids:
        brier_loss = (outcome - b.prob_violation) ** 2
        reward = alpha * ((1.0 - brier_loss) - 0.75)
        vcg_charge = auction_result.clarke_prices.get(b.verifier_id, 0.0) if b.verifier_id in winner_set else 0.0
        delta = reward - vcg_charge
        new_bankroll = max(0.0, bankrolls[b.verifier_id] + delta)
        settlements.append(BrierSettlement(
            verifier_id=b.verifier_id,
            brier_loss=brier_loss,
            delta_bankroll=delta,
            new_bankroll=new_bankroll,
        ))
    return settlements


def update_calibration_weights(
    loss_histories: dict[str, list[float]],
) -> dict[str, float]:
    """w_i = max(0.1, 1 - mean(losses))"""
    weights = {}
    for vid, losses in loss_histories.items():
        if not losses:
            weights[vid] = 1.0
        else:
            weights[vid] = max(0.1, 1.0 - sum(losses) / len(losses))
    return weights
