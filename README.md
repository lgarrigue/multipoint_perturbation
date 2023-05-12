This code contains the structures and functions enabling to create and apply multipoint perturbation up to second order and standard perturbation up to third order, to periodic Schrödinger operators. It produces the plots of the article **A multipoint perturbation formula for eigenvalue problems**, written by Benjamin Stamm, Louis Garrigue and Geneviève Dusson.

## How to run the code
Open Julia with
```
julia
```
and write
```
include("applications_producing_plots_of_article.jl")
```

## Libraries
The libraries used are
- for functions implementing multipoint perturbation : LinearAlgebra, MKL, Optim
- for using Schrödinger operators : FFTW
- for plotting functions : CairoMakie, LaTeXStrings

One may have problems with the used version of LaTeXStrings. Moreover, we use the version 1.9 of Julia
