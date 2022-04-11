# Tenta plugg

## Gradient descent

### Batch GD

### Stokastisk GD

### Mini batch GD

## PCA principal component analysis

- Preprocessing of data to reduce dimension to avoid curse of dimensionality
- Lågdimensionell representation av datasetet
- Linjärkombination av feature variabler med maximal varians med hjälp av hyperplan

0. feature standardize data
1. skapa kovariansmatris av X(feature space)
2. beräkna egenvektorer
3. sortera egenvektorer efter egenvärdens storlek
4. välj antal(d) dimensioner (components)
5. skapa ett ortogonalt system av components och projicera alla punkter på det systemet

Man väljer d dimensioner genom att göra en elbow plot på proportion variance, sen kan man i sin pipeline välja antalet d genom att ta de d som är innan elbowen

## Unsupervised learning

What to learn more specific here?

### K-means

- finns ej labels och är därför mer subjektiv vad som är korrekt och inte
- clustering / gruppera data

1. välj antalet(k) kluster
2. kluster center väljs godtyckligt
3. närmaste punkterna klassificeras
4. beräkna nytt kluster center genom att titta på tyngdpunkten av varje klass av punkter och placera det nya kluster centret där
5. upprepa 3 och 4 tills inga nya reassignments sker

Man väljer antalet kluster genom att göra en elbow plot av sum of least squared, sedan väljer man antalet kluster där derivatan är som lägst, för att bli säkrare på sitt val så kan man göra en silhouette score elbow plot för att se hur tighta kluster för vid varje k kluster

## ANN

What to learn?
