"""Feature extractors. A method = one of these on the shared pipeline spine.

    saplma   -> mean over output-token hidden states at a chosen layer
    ptrue    -> hidden state at an appended "is this true?" verdict position
    lookback -> mean attention lookback ratio (context vs own output) over all heads
"""
