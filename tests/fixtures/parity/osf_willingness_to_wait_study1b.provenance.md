OSF Willingness-to-Wait study1b fixture

This slim CSV was copied from the sibling mixeff wrapper fixture:

`/Users/bbuchsbaum/code/mixeff/tests/fixtures/osf_willingness_to_wait_study1b.csv`

The wrapper fixture is reconstructed from OSF node `ftexh`, file `3bxkt`
(`FullData_replication.csv`), and retains the trial-level modeling columns used
by the published lme4 script:

`ID`, `Title`, `wait_choice`, `Enjoyment`, `arousal`, `Q1_correct`,
`Q2_correct`, and `SVScore`.

Rust tests center `Enjoyment` locally before fitting the study1b crossed
binomial GLMM:

`wait_choice ~ 1 + Enjoyment_centered + (1 + Enjoyment_centered | ID) + (1 + Enjoyment_centered | Title)`
