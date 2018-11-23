# Black Friday
## Predicting Black Friday purchase amounts from customer data

If you look at when this repository went up it's entirely topical. Anyhow,
here are the data:

https://www.kaggle.com/sdolezel/black-friday

There are a mixture of numerical and categorical data intended to predict
sales volumes for individuals shopping on Black Friday (and most of them are
spending a pretty penny, with a median value of $8062). I have chosen to use
the Kaggle favorite XGBoost library for this task along with scikit-learn's
`RandomizedSearchCV` to improve results. Something that surprised me was the
fact that the product and customer ID were actually meaningful in the task,
which I discovered after looking for results others had gotten to compare them
with my own. The dataset was the subject of a [hackathon hosted by Analytics
Vidhya](https://medium.com/data-science-analytics/black-friday-data-science-hackathon-4172a0554944).
Adding them to the model significantly boosted results. As with the earlier
[cover type
repository](https://github.com/lehighvalley-schoolofai/cover-type), the whole
process can be carried out by running `./run.sh`.

After running the solution presented here overnight I was greeted upon waking
up with the following results:

```
SD of y_test: 4987.95
RMSE of y_pred: 2618.30
```

The standard deviation is identical to the RMSE if the mean is predicted and
my results cut that RMSE nearly in half, a substantial improvement over this
benchmark. Is it the best imaginable? No, because the lowest RMSE in the
Analytics Vidhya hackathon was about 2408. Nonetheless, the answer achieved
here is only about 9% larger than this best outcome and was achieved without
any use of deep learning or ensembles (beyond what XGBoost provides itself)
and uses no very sophisticated feature engineering. Moreover, the outcome is
much closer to the top entry than the benchmark of guessing the mean and so,
given the circumstances, can certainly be considered a win. So, always bear in
mind the trade-off between better results and more effort and the notion of
diminishing marginal returns as well.
