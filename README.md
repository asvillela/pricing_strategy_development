# pricing_strategy_development
Identify the pricing strategy against key competitors to optimize profit, based on sales and pricing data.

This code was used to redesign and automatize the pricing strategy process for one of P&G Categories in a specific Region.
It leverages Nielsen sales and pricing data and internal financial data from skus.

For confidentiality purposes, the code shared here does not include any real P&G data. It uses a Kaggle database (link below) and dummy financial data.
https://www.kaggle.com/datasets/bhanupratapbiswas/retail-price-optimization-case-study

The process includes the following steps:

IDENTIFY ANCHORS
Anchors are the own skus we are building the pricing strategy for.
Objective is to identify the skus that will drive the most value. The tool helps identify the highest-selling skus, but other strategic skus may be added as needed.

IDENTIFY PAIRS
Pairs are the competitive skus that have the highest impact on anchor sales.
Leveraging sales data, we identify the pair with the highest negative correlation between sales and the price-index (anchor vs pair).

IDENTIFY OPTIMAL PRICE INDEX
Modeling the sales (or shares) per price-index vs pair, and including internal financial structure of anchor skus, we can identify the price-index vs pair that will result in highest sales or profit.
