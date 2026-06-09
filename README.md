# Analysis of Juvenile Gender-wise Crime Rates and Total Crime Rates using Data Preprocessing Techniques

*By Kavin Kumaran*
*MAY 27, 2026*

---

## Abstract

This project demonstrates the application of data preprocessing techniques to analyse the relationship between juvenile gender-wise crime rates and total gender-wise crime rates, with the objective of determining whether juvenile crime rates can serve as indicators of total crime rates for an arbitrary city. Various preprocessing methods were applied to improve the quality and consistency of the dataset, enabling more reliable analysis and model development. Preliminary analyses were performed to identify correlations, trends, and relationships among the target and predictor variables. The preprocessing steps included handling inconsistencies, removing redundant information, performing feature selection, and scaling the data where necessary. The resulting processed dataset is more suitable for machine learning applications and further predictive studies.

---

## Introduction

### Importance of Crime Data Analysis

It is important to analyse crime data to better understand the causes of crime and possible solutions to reduce it. In this report, we form the hypothesis that juvenile crime rates are highly correlated with total crime rates in a city. If we are able to verify this hypothesis, we can formulate further experiments to check if reducing juvenile crime can lead to a fall in total crime rates.

### Requirement of Data Preprocessing

Real world data often contains outliers, impossible and infeasible entries, invalid datatypes, missing data, etc. These may be due to human error, random noise, and low accuracy measurements. Additionally, data often contains features of different scales, and categorical variables that have no direct numerical correlation. To avoid the effect of inaccuracies these sources may add, possibly causing a breakdown of model accuracy, to encode categorical variables, and to avoid feature negligence due to scale, we performed preprocessing on the acquired real-world data.

### Project Outcome

To successfully implement modern data-preprocessing techniques to real-world crime statistics, so that they can further be used for model training, data analysis, and ultimately test the hypothesis that juvenile gender-wise crime rates are responsible for overall gender-wise crime rates.

---

## Dataset

| Attribute | Description |
|---|---|
| S.No | Serial number |
| Cities | City name |
| Male Juveniles | Juvenile male crime count |
| Female Juveniles | Juvenile female crime count |
| Total Juveniles | Juvenile crimes across all genders |
| Male Total | Total male crimes |
| Female Total | Total female crimes |
| Grand Total | Total crimes across all genders |

| Property | Value |
|---|---|
| Number of Rows | 20 |
| Total Number of Columns | 20 |
| Number of Columns Used | 6 |

---

## Exploratory Data Analysis

Initial inspection revealed non-uniform spacing and names in feature variables (`data.columns`). Column names were standardised to uniform snake_case format.

**`data.info()` output:**

```
<class 'pandas.DataFrame'>
RangeIndex: 20 entries, 0 to 19
Data columns (total 20 columns):
 #   Column             Non-Null Count  Dtype
---  ------             --------------  -----
 0   S_No               20 non-null     str
 1   City               20 non-null     str
 2   Male_juveniles     20 non-null     int64
 3   Female_juveniles   20 non-null     int64
 4   Total_juveniles    20 non-null     int64
 5   Male_18to30        20 non-null     int64
 6   Female_18to30      20 non-null     int64
 7   Total_18to30       20 non-null     int64
 8   Male_30to45        20 non-null     int64
 9   Female_30to45      20 non-null     int64
10   Total_30to45       20 non-null     int64
11   Male_45to60        20 non-null     int64
12   Female_45to60      20 non-null     int64
13   Total_45to60       20 non-null     int64
14   Male_60plus        20 non-null     int64
15   Female_60plus      20 non-null     int64
16   Total_60plus       20 non-null     int64
17   Male_total         20 non-null     int64
18   Female_total       20 non-null     int64
19   Grand_total        20 non-null     int64
dtypes: int64(18), str(2)
memory usage: 3.6 KB
```

**`data.describe()` summary statistics** (selected columns):

| Stat | Male_juveniles | Female_juveniles | Total_juveniles | Male_total | Female_total | Grand_total |
|---|---|---|---|---|---|---|
| count | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 |
| mean | 779.0 | 1.6 | 780.6 | 20142.2 | 843.2 | 20985.4 |
| std | 1782.6 | 3.72 | 1785.9 | 44281.9 | 1837.6 | 46076.2 |
| min | 0.0 | 0.0 | 0.0 | 1192.0 | 0.0 | 1197.0 |
| 25% | 24.25 | 0.0 | 24.25 | 4808.5 | 51.5 | 4936.25 |
| 50% | 286.5 | 0.0 | 289.5 | 7201.5 | 320.0 | 8003.0 |
| 75% | 486.75 | 2.0 | 488.75 | 13612.0 | 963.25 | 14325.5 |
| max | 7790.0 | 16.0 | 7806.0 | 201422.0 | 8432.0 | 209854.0 |

**Raw data (`data`) — City-level crime counts:**

| S_No | City | Male_juveniles | Female_juveniles | Total_juveniles | Male_total | Female_total | Grand_total |
|---|---|---|---|---|---|---|---|
| 1 | Ahmedabad (Gujarat) | 298 | 0 | 298 | 31898 | 3054 | 34952 |
| 2 | Bengaluru (Karnataka) | 197 | 3 | 200 | 26337 | 2021 | 28358 |
| 3 | Chennai (Tamil Nadu) | 816 | 0 | 816 | 53804 | 4704 | 58508 |
| 4 | Coimbatore (Tamil Nadu) | 35 | 0 | 35 | 6940 | 1032 | 7972 |
| 5 | Delhi | 3073 | 2 | 3075 | 96959 | 2127 | 99086 |
| 6 | Ghaziabad (Uttar Pradesh) | 0 | 0 | 0 | 4270 | 7 | 4277 |
| 7 | Hyderabad (Telangana) | 377 | 0 | 377 | 7890 | 310 | 8200 |
| 8 | Indore (Madhya Pradesh) | 275 | 6 | 281 | 9231 | 537 | 9768 |
| 9 | Jaipur (Rajasthan) | 404 | 1 | 405 | 15109 | 1235 | 16344 |
| 10 | Kanpur (Uttar Pradesh) | 9 | 0 | 9 | 8349 | 76 | 8425 |
| 11 | Kochi (Kerala) | 15 | 0 | 15 | 14930 | 279 | 15209 |
| 12 | Kolkata (West Bengal) | 19 | 0 | 19 | 12249 | 2243 | 14492 |
| 13 | Kozhikode (Kerala) | 26 | 0 | 26 | 4087 | 171 | 4258 |
| 14 | Lucknow (Uttar Pradesh) | 80 | 0 | 80 | 8272 | 0 | 8272 |
| 15 | Mumbai (Maharashtra) | 668 | 0 | 668 | 51312 | 3231 | 54543 |
| 16 | Nagpur (Maharashtra) | 442 | 2 | 444 | 10614 | 464 | 11078 |
| 17 | Patna (Bihar) | 15 | 0 | 15 | 8236 | 48 | 8284 |
| 18 | Pune (Maharashtra) | 420 | 0 | 420 | 15985 | 1138 | 17123 |
| 19 | Surat (Gujarat) | 621 | 2 | 623 | 24378 | 1242 | 25620 |
| 20 | **TOTAL CITIES** | **7790** | **16** | **7806** | **410850** | **23919** | **434769** |

---

## Visualisations

Three plots were generated:

1. **Scatter plot** — `Male_juveniles` vs `Male_total`: Suggests a positive trend with one high-leverage point (Delhi).
2. **Scatter plot** — `Female_juveniles` vs `Female_total`: Sparse distribution; female juvenile counts are very low (0–16 range).
3. **Stacked bar chart** — `Male_total` and `Female_total` by city, sorted descending by `Grand_total`: Delhi, Chennai, and Mumbai dominate. The `TOTAL CITIES` aggregate row is visually apparent as an outlier.

```python
# Scatter plots
sns.scatterplot(x="Male_juveniles", y="Male_total", data=data)
sns.scatterplot(x="Female_juveniles", y="Female_total", data=data)

# Stacked bar chart
sorted_data = data.sort_values(by="Grand_total", ascending=False)
sorted_data.set_index("City")[["Male_total", "Female_total"]].plot(
    kind="bar",
    stacked=True
)
plt.xticks(rotation=90, ha='center')
plt.tight_layout()
plt.show()
```

---

## Preprocessing Steps

### 1. Duplicate Removal

Absence of duplicated rows was verified. Hence, no further action was taken.

```python
n_duplicate_rows = data.duplicated().sum()
print(f'There are {n_duplicate_rows} duplicate rows.')
if n_duplicate_rows == 0:
    print('No removal of duplicates required')
else:
    print('Must remove duplicates')
```

```
There are 0 duplicate rows.
No removal of duplicates required
```

### 2. Imputing

Absence of missing values was verified. Hence, no further action was taken.

```python
data.isna().sum()
```

```
S_No                0
City                0
Male_juveniles      0
Female_juveniles    0
Total_juveniles     0
Male_18to30         0
Female_18to30       0
Total_18to30        0
Male_30to45         0
Female_30to45       0
Total_30to45        0
Male_45to60         0
Female_45to60       0
Total_45to60        0
Male_60plus         0
Female_60plus       0
Total_60plus        0
Male_total          0
Female_total        0
Grand_total         0
dtype: int64

No empty/invalid values to impute
```

### 3. Dropping Outliers

From the stacked bar plot on the Visualisations page, it is apparent that the `TOTAL CITIES` row has a much greater scale. It is an outlier and is hence removed from the dataset.

### 4. Feature Selection

Since the aim of this project is to analyse the relationship between juvenile gender-wise crime rates and total gender-wise crime rates, we ignored the categorical variables `Cities` and `S.no`. We kept only the columns corresponding to juvenile gender-wise crime rates and total gender-wise crime rates and dropped the rest. Subsequently, we measured the correlation between the feature variables and found strong correlation between Male and Total juveniles. Hence, we drop `Total_juveniles` to avoid redundancy.

**Correlation matrix** (selected columns):

| | Male_juveniles | Female_juveniles | Total_juveniles | Male_total | Female_total | Grand_total |
|---|---|---|---|---|---|---|
| **Male_juveniles** | 1.000000 | 0.222180 | 0.999998 | 0.903306 | 0.384049 | 0.889881 |
| **Female_juveniles** | 0.222180 | 1.000000 | 0.224337 | 0.102306 | −0.052183 | 0.095551 |
| **Total_juveniles** | 0.999998 | 0.224337 | 1.000000 | 0.903080 | 0.383736 | 0.889647 |
| **Male_total** | 0.903306 | 0.102306 | 0.903080 | 1.000000 | 0.682171 | 0.999199 |
| **Female_total** | 0.384049 | −0.052183 | 0.383736 | 0.682171 | 1.000000 | 0.710876 |
| **Grand_total** | 0.889881 | 0.095551 | 0.889647 | 0.999199 | 0.710876 | 1.000000 |

**We are left with:**

```python
X_cols = ['Male_juveniles', 'Female_juveniles']
y_cols = ['Male_total', 'Female_total', 'Grand_total']
X = data.loc[:, X_cols].to_numpy()
y = data.loc[:, y_cols].to_numpy()
print(f'Shape of X: {X.shape}\nShape of y: {y.shape}')
```

```
Shape of X: (19, 2)
Shape of y: (19, 3)
```

### 5. Feature Scaling

From the axes of the scatterplots in the Visualisations section, it is apparent that the different features possess a variety in scale. This may disrupt the accuracy of models, and hence feature scaling is performed. We implemented the `MinMaxScaler` and `StandardScaler`. On further analysis, one of these may be selected based on the best accuracy.

---

## Final Observations

1. Female crime rates are significantly lower than male crime rates.
2. There is high correlation between male juvenile crime rates and total male crime rates.
3. There is high correlation between male juvenile crime rates and total juvenile crime rates.
4. There is poor correlation between male and female crime rates across different age groups.
5. There is poor correlation between female juvenile crime and female total crime.
6. Crime in each city is not uniform, with Delhi, Chennai, and Mumbai having the highest crime rates.
7. The data was fairly robust, with no missing, duplicated, or broken datapoints.
8. Unnecessary and redundant columns were dropped.

---

## Conclusion

Data preprocessing significantly improved the quality of the dataset. The transformed dataset is cleaner, more consistent, and suitable for machine learning and predictive analysis.

---

## References

- **Data.gov.in** — Open Government Data Platform India: Source of the crime statistics dataset used in this project.
- **Pandas Documentation**: Used for data manipulation, preprocessing, and analysis.
- **NumPy Documentation**: Used for numerical operations and array processing.
- **Matplotlib Documentation**: Used for data visualization.
- **Seaborn Documentation**: Used for statistical plotting and visualization.
- **Scikit-learn Documentation**: Used for preprocessing methods including feature scaling.
