# Task 2: Exploratory Data Analysis of Amazon Fine Food Reviews

## Project Overview

This project focuses on exploratory data analysis (EDA) of the Amazon Fine Food Reviews dataset using Python. The objective is to examine the dataset's structure, assess data quality, summarize customer ratings, identify review patterns, and investigate selected statistical relationships.

The analysis demonstrates practical skills in data inspection, data quality assessment, descriptive statistics, and hypothesis testing.

## Objectives

- Examine the dataset's dimensions, variables, data types, and missing values.
- Identify potential data quality issues, including repeated records and invalid helpfulness values.
- Analyze customer rating distributions and review characteristics.
- Investigate selected statistical relationships using hypothesis tests.
- Produce structured reports and summary tables to communicate the findings.

## Dataset Description

The dataset contains Amazon customer reviews of fine food products.

The analysis identified the following dataset characteristics:

| Attribute | Result |
|---|---:|
| Total records | 568,454 |
| Total columns | 10 |
| Unique products | 74,258 |
| Unique users | 256,059 |
| Missing `ProfileName` values | 26 |
| Missing `Summary` values | 27 |
| Duplicate IDs | 0 |
| Exact duplicate rows | 0 |

**Dataset source:** [Amazon Fine Food Reviews — Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

The original dataset is not included in this repository because of its large file size. Download `Reviews.csv` from the source above before running the analysis.

## Tools and Technologies Used

- **Python** — analysis and automation.
- **Pandas** — data loading, inspection, summarization, and quality checks.
- **NumPy** — numerical operations, where applicable.
- **SciPy** — statistical hypothesis testing, where applicable.
- **Matplotlib and Seaborn** — data visualization, where applicable.

## Methodology

### 1. Data Inspection
Examined the dataset's shape, column names, data types, unique values, and missing values.

### 2. Data Quality Assessment
Checked for duplicate IDs, exact duplicate rows, repeated review keys, invalid ratings, and inconsistencies in helpfulness counts.

The checks identified 1,205 records sharing the examined combination of product, user, timestamp, and review text, as well as two records where the helpfulness numerator exceeded the denominator.

These findings were flagged for review rather than automatically removed.

### 3. Descriptive Analysis
Summarized customer ratings, product and user counts, review dates, and review text lengths.

### 4. Statistical Hypothesis Testing
Conducted selected statistical tests to investigate rating patterns, differences between reviews with and without helpfulness votes, and review-text length differences between one-star and five-star reviews.

Statistical significance was interpreted alongside practical significance and the limitations of the data.

## Key Findings

- The average customer rating was approximately **4.18 out of 5**.
- The median rating was **5 stars**.
- Approximately **63.88% of reviews received 5 stars**.
- The dataset contained **74,258 unique products** and **256,059 unique users**.
- Review timestamps ranged from October 1999 to October 2012.
- The data quality checks identified potential inconsistencies that warrant further investigation.
- Statistical tests identified differences in selected review characteristics. These associations do not establish causation.

## Project Files

The Task 2 folder contains the Python analysis script and the outputs produced by the analysis, including:

- `reviews_task2_task3_analysis.py` — analysis script, if this is the script used for Task 2.
- `analysis_report.txt` — summary of the analysis and data quality findings.
- CSV files containing variable information, data quality flags, and hypothesis-test results.

File names may differ depending on the version of the script used.

## How to Run the Project

### Prerequisites
Install Python 3 and the required libraries.

```bash
python -m pip install pandas numpy scipy matplotlib seaborn
```

### Download the Dataset
1. Download the Amazon Fine Food Reviews dataset from Kaggle.
2. Extract `Reviews.csv`.
3. Place the CSV file in the location expected by the Python script.

### Execute the Analysis

Open a terminal in the project directory and run:

```bash
python reviews_task2_task3_analysis.py
```

The script should generate its analysis outputs in the configured output directory. If your actual Task 2 script has a different name, replace the filename in the command above.

## Limitations

- The dataset covers a historical period and may not represent current customer preferences.
- Customer ratings are ordinal; averages should be interpreted with that limitation in mind.
- Statistical significance does not necessarily imply a large or practically important effect.
- Review text length is not a direct measure of sentiment.
- Flagged records require further investigation before deciding whether to remove or modify them.

## Conclusion

This project demonstrates the use of Python for exploratory data analysis, data quality assessment, descriptive statistics, and hypothesis testing. The results provide an overview of customer rating patterns and selected review characteristics while highlighting data quality considerations that should be addressed before more advanced analysis.

## Author

**Iguoba Victor**

Computer Engineering Student | Aspiring Data Analyst
