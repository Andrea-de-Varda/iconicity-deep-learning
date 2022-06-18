# Iconicity and deep learning
Code employed in the deep learning-based study of iconicity in language. Details can be found in the paper (de Varda &amp; Strapparava (2022). [A Cross‐Modal and Cross‐lingual Study of Iconicity in Language: Insights From Deep Learning.](https://onlinelibrary.wiley.com/doi/full/10.1111/cogs.13147) Cognitive Science.)

### Experiment data
We release the cosine similarity between target and predicted vectors (Exp 1: visual vectors, Exp. 2: semantic vectors), which can be considered as data-driven iconicity measurements. They reflect how well the model was able to infer the visual and semantic representation of a word from its phonetic representation, after being trained in the other languages. For comparison, we also report in the dataframes the iconicity ratings for the same words obtained by [Winter et al., 2022](https://osf.io/qvw6u/) and [Winter et al., 2017](https://www.researchgate.net/publication/318364562_Which_words_are_most_iconic_Iconicity_in_English_sensory_words).

### Additional data
The data on which our experiments were performed can be found here:
- [THINGS database](https://things-initiative.org/)
- [Word vectors](https://www.marekrei.com/projects/vectorsets/)  (we employed the ones derived with Word2Vec, with window size = 100)
- [PoS-tagged BNC lexicon](http://www.kilgarriff.co.uk/bnc-readme.html#raw)
