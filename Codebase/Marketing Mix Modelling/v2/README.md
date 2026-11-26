# Introduction 

Marketing Mix Modeling (MMM) is a regression-based analytical technique used to measure the effectiveness of marketing campaigns. It helps quantify how different marketing activities and spend levels impact key outcomes like attendance or transactions. The goal is to guide smarter budget allocation by identifying which campaigns and business lines drive the highest return on investment (ROI).

# Getting Started

## Environment Setup

```
%pip install -r ../requirements.txt
%restart python
```

To read and export the data in Databricks, create a volume and upload all the files located in `data/`.

## Software Dependencies

- Python 3.10+ is required for full functionality of PyMC Marketing
- If the model is run on Databricks, ensure runtime of 13.3+ LTS

# Build and Test

The example notebook used to build this model can be found [here.](https://www.pymc-marketing.io/en/latest/notebooks/mmm/mmm_example.html)

# Future Improvements

- Rules for raw data (common variables, LOBs, etc.)
- More queries to Databricks tables
- Less CSV files
- Model priors specified using research or domain knowledge
- Adstock and saturation parameters defined for each LOB/channel
- PyMC Marketing budgeting tool used for scenario analysis
- PyMC Marketing lift test used for scenario analysis
