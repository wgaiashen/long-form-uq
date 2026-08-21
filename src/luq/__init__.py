"""luq — long-form uncertainty quantification pipeline.

The project is four shared stages and a method is just a choice of feature extractor:

    data  ->  generate (+cache)  ->  features  ->  label  ->  probe  ->  PRR

Build the stages once; SAPLMA, P(True), and later Lookback Lens differ only in
`luq.features.*`. See the project's working notes for the architecture and cache design.
"""
