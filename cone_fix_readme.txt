to make a cone plot, plotly goes through each vector and projects it onto the
distance to the next vector (in order of the dataframe), and then scales
all vectors based on the smallest projection it finds. This means all
arrays will have arbitrary cone sizes, in particular some extremely
small if all cones perpendicular to their displacement

to fix this, add in two very small dummy vectors very close together,
such that the distance between them is the smallest projection for
all protein arrays. This is constant between all arrays, so plotly
will always choose this as the scaling factor, thus scaling is constant
between every array.

see https://github.com/plotly/plotly.js/issues/3613
