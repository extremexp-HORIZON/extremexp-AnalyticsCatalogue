# Randomly constrained learning
This library consists of a class interface for training and benchmarking neural
networks that meet dynamically varying user needs.

::include{file=md/overview.md}

## Components
::include{file=md/components.md}
## Using the library
::include{file=md/installation.md}
## Costs
::include{file=md/cost.md}
## Incomplete TODO

- Correct/varied loss functions
- Allow benchmarking using statistics that reflect cost uncertainty, rather
than calculating fixed costs - that is really the point of these losses -
need to ensure the motivation for these scores is well understood.
- Systematically understand role of loss, avoid ad hoc searches where possible
- - this is a rescaling of the last layer, why does it matter?
- - theory doesn't align precisely
- Systematically apply k means
- Visualisation
- Deal with multiple parametric losses and architectures
- Clear documentation - pdoc or similar as a reference/use notebooks
- Testing, type checking - some NamedTuple classes could be
possibly better dealt with using numpy Annotations etc
