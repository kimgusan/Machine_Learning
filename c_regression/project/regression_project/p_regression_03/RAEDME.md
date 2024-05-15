# ③ Regression03 (다차원)

## 주제: # 🏥 뉴욕주립병원 입원환자 퇴원 금액

    (1) 데이터 원본: ex) https://kaggle.com)

### 목차

1. **가설 설정**
2. **데이터 분석**
3. **데이터 전처리**
4. **데이터 훈련**
    <details>
        <summary>Cycle</summary>   
        <ul style='list-style-type: none;'>
            <li><a href="#cycle01">Cycle01(전처리 이후 회귀 훈련)</a></li>
            <li><a href='#cycle02'>Cycle02(다중공선성 및 상관관계 확인 후 분석 효율을 위한 차원 축소 진행)</a></li>
            <li><a href='#cycle03'>Cycle03(모델의 신뢰성을 위한 교차검증 진행)</a></li>
        </ul>
   </details>
5. **결론**

## 1. 가설 설정

### 가설 1: 의료 서비스 상관관계 분석

-   **퇴원 금액 상관관계**: 병원 퇴원 시 측정되는 금액과 다양한 요소들 간의 상관관계를 분석하여, 금액에 대한 회귀 분석을 수행합니다.
-   **응용 가능성**:
    -   **진단 코드와 의료 보장**: 필요한 의료 보장이 더 많이 필요한 지역을 진단하고, 이를 통해 의료 서비스의 효율성을 높입니다.
    -   **비용 효율성과 지불 소스 분석**: 병원은 다양한 지불 소스 간의 상관 관계 분석을 통해 비용 효율성을 향상시킬 수 있습니다.

<hr>

### 2. 데이터 분석

```
import pandas as pd

h_df = pd.read_csv('../../../datasets/p_hospital-inpatient-discharges-sparcs-de-identified-2010-1.csv', low_memory=False)
h_d

h_df.info()
```

### 3. 데이터 전처리

<details>
  <summary>Click data preprocessing</summary>

```
# 결측치 확인
h_df.isna().sum()

# 중복값 확인
h_df.duplicated().sum()

# 불필요 컬럼 삭제
columns = ['index','Health Service Area','Hospital County','Operating Certificate Number', 'Facility ID','Age Group', 'Gender', 'Race',
'Ethnicity','Length of Stay', 'Type of Admission', 'Patient Disposition','Discharge Year', 'CCS Diagnosis Code','CCS Procedure Code',
'APR DRG Code', 'APR MDC Code','APR Severity of Illness Code','APR Risk of Mortality','APR Medical Surgical Description','Attending Provider License Number',
'Operating Provider License Number','Other Provider License Number','Birth Weight','Abortion Edit Indicator',
'Emergency Department Indicator', 'Discharge Year', 'Total Charges']

pre_h_df = h_df[columns].copy()
pre_h_df

pre_h_df = pre_h_df.drop(labels = ['index','Discharge Year', 'Abortion Edit Indicator'], axis =1)

# 결측치 삭제
pre_h_df = pre_h_df.dropna().reset_index(drop=True)

# 중복값 삭제
pre_h_df = pre_h_df.drop_duplicates().reset_index(drop=True)


# 정보 확인 후 범주형 데이터 분리
pre_h_df.info()


# 범주형 데이터 분리

category_h_df = pre_h_df.select_dtypes(include=['object']).copy()

# int, float 데이터 분리
numeric_h_df = pre_h_df.select_dtypes(include=['int64', 'float64']).copy()


# Label encoding
from sklearn.preprocessing import LabelEncoder

columns = category_h_df.columns
encoders = {}

for column in columns:
    encoder = LabelEncoder()
    # 각 컬럼 데이터를 리스트로 변환
    category_h_df[column] = encoder.fit_transform(category_h_df[column].tolist())
    encoders[column] = encoder.classes_

category_h_df.sort_index(inplace=True)
numeric_h_df.sort_index(inplace=True)

# # 범주형과 수치형 데이터 프레임을 결합합니다.
num_h_df = pd.concat([category_h_df, numeric_h_df], axis=1)
num_h_df



# 이상치 삭제 진행을 위한 정규화 작업
from sklearn.preprocessing import StandardScaler

std = StandardScaler()
result = std.fit_transform(num_h_df)
std_num_h_df = pd.DataFrame(result, columns=num_h_df.columns)
std_num_h_df

condition = True
error_count = []

for column in std_num_h_df.columns:
    # 현재 컬럼에 대해 -1.96과 1.96 사이에 속하는 값을 카운트합니다.
    count = std_num_h_df[column].between(-1.96, 1.96).sum()
    error_count.append(count)
    condition &= std_num_h_df[column].between(-1.96, 1.96)

std_num_h_df = std_num_h_df[condition]
std_num_h_df

for column, count in zip(std_num_h_df.columns, error_count):
    # 이상치의 개수는 전체 데이터 개수에서 정상적인 값의 개수를 빼면 됩니다.
    outlier_count = len(std_num_h_df) - count
    outlier_ratio = (outlier_count / len(std_num_h_df)) * 100
    print(f"'{column}'에 대한 이상치 개수: {outlier_count},\n {column}에 대한 이상치 비율'{round(outlier_ratio,2)}%'\n")


# 이상치 제거
condition = True
error_count = []

for column in std_num_h_df.columns:
    if std_num_h_df[column].between(-1.96, 1.96) is True:
        error_count.append(std_num_h_df[column].between(-1.96, 1.96).count())
    condition &= std_num_h_df[column].between(-1.96, 1.96)

std_num_h_df = std_num_h_df[condition]
std_num_h_df


# 이상치 제거한 데이터를 인덱스 번호에 맞게 가져오기
numeric_h_df = numeric_h_df.iloc[std_num_h_df.index].reset_index(drop=True)
numeric_h_df

num_h_df['Total Charges'] = np.log1p(num_h_df['Total Charges'])

```

</details>

    
<img width="540" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/7aa3e2e7-1be9-4180-8799-cded1a13e522">
<img width="398" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/d5bad4b0-6e48-4e06-bf09-50c0f028460a">

### 4. 데이터 훈련

<h2 id='cycle01'>Cycle01</h2>
<p>1. 타겟 데이터 분포가 일정하여 차원 축소 없이 분석 진행</p>


<details>
  <summary>Click Cycle01_code</summary>
    
```
# 회귀 분석 모델 사용
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

features, targets = num_h_df.iloc[:, :-1], num_h_df.iloc[:, -1]

X_train, X_test, y_train, y_test = \
train_test_split(features, targets, test_size=0.2, random_state=321)

l_r = LinearRegression()
l_r.fit(X_train, y_train)

prediction = l_r.predict(X_test)
get_evaluation(y_test, prediction)

MSE: 0.4996, RMSE: 0.7068, MSLE: 0.0041, RMSLE: 0.0644, R2: 0.5210
```
<img width="354" alt="스크린샷 2024-05-15 오후 11 11 31" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/357d68c7-4630-4822-943c-b4659ab4b934">

```
# 비선형 모델 사용
# 회귀 분석 모델 사용
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2).fit_transform(features)

X_train, X_test, y_train, y_test = \
train_test_split(poly_features, targets, test_size=0.2, random_state=321)

l_r = LinearRegression()
l_r.fit(X_train, y_train)

prediction = l_r.predict(X_test)
get_evaluation_negative(y_test, prediction)

MSE: 0.3935, RMSE: 0.6273, R2: 0.6228
```
<img width="225" alt="스크린샷 2024-05-15 오후 11 11 34" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/5f0e53fe-c4d4-4653-a689-63bd9a20a650">


```
# 트리 모델 훈련
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

features, targets = num_h_df.iloc[:, :-1],  num_h_df.iloc[:, -1]

X_train, X_test, y_train, y_test =\
train_test_split(features, targets, test_size=0.2, random_state=321)

dt_r = DecisionTreeRegressor(random_state=321)
rd_r = RandomForestRegressor(random_state=321)
gb_r = GradientBoostingRegressor(random_state=321)
xgb_r = XGBRegressor(random_state=321)
lgb_r = LGBMRegressor(random_state=321)

models = [dt_r, rd_r, gb_r, xgb_r, lgb_r]

for model in models:
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print(model.__class__.__name__)
    get_evaluation(y_test, prediction)
```
</details>

<img width="546" alt="스크린샷 2024-05-15 오후 11 11 43" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/3dd0457f-53ec-40cc-a9fe-0e3b2d5cde6e">
<img width="551" alt="스크린샷 2024-05-15 오후 11 11 49" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/a5fd267a-bf7c-45d2-bc75-f93a11d139fa">

    
<h2 id='cycle02'>Cycle02</h2>
<p>1. 모델의 공분산성을 지닌 수치 중 높은 수치에 대하여 삭제 후 확인</p>
<p>2. 모델 훈련속도의 효율을 높이기 위해 차원축소를 진행.</p>

<details>
  <summary>Click Cycle02_code</summary>

```
# 상관관계 확인
num_h_df.corr()['Total Charges'].sort_values(ascending=False)[1:]

import seaborn as sns
corr = num_h_df.corr()
sns.heatmap(corr, cmap='Oranges')

from statsmodels.stats.outliers_influence import variance_inflation_factor

def get_vif(features):
    vif = pd.DataFrame()
    vif['vif_score'] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
    vif['feature'] = features.columns
    return vif

get_vif(features)

# 불필요 feature 제거
c2_h_df = num_h_df.drop(labels = ['Operating Certificate Number', 'APR DRG Code', 'Health Service Area'], axis = 1)

```

<h4>before</h4>
<img width="512" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/f701ff24-270c-4292-8c26-4916fc217b57">
<img width="242" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/afeb8d47-ed45-42cc-9df3-bd681999e1c4">

<h4>after</h4>
<img width="230" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/7ed00221-7092-4635-9bc4-f394616364b1">
<img width="250" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/3facd9dd-8929-4cc2-b45a-07c78d123eb3">

```
# 차원축소 진행
from sklearn.model_selection import train_test_split

features, targets = c2_h_df.iloc[:, :-1], c2_h_df.iloc[:, -1]

X_train, X_test, y_train, y_test = \
train_test_split(features, targets, test_size=0.2, random_state=124)


# 보존율 확인
from sklearn.decomposition import PCA

for i in range(4):
    pca = PCA(n_components=(i + 1))

    pca_train = pca.fit_transform(X_train)

    # 보존율
    print(pca.explained_variance_ratio_.sum())
```
<img width="118" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/62101d32-56d4-4ece-a34d-30278d2e7c7d">

```
# 차원 축소 후 최적의 모델을 사용하여 분석 진행
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

features, targets = c2_h_df.iloc[:, :-1], c2_h_df.iloc[:, -1]

X_train, X_test, y_train, y_test = \
train_test_split(features, targets, test_size=0.2, random_state=124)

l_r = LinearRegression()

# pipe = Pipeline([('pca', PCA(n_components=2)), ('l_r', l_r)])
# 성능이 가장 좋았던 모델로 사용
pipe = Pipeline([('pca', PCA(n_components=8)), ('lgb_r', LGBMRegressor(random_state=321))])
pipe.fit(X_train, y_train)

prediction = pipe.predict(X_test)
get_evaluation_negative(y_test, prediction)

```
<img width="105" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/3620c683-d2ee-4324-aae4-d70c03ceac4c">
<img width="208" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/f92a5aad-62d5-4330-80fe-9ef57af55102">

```
# train, validation 그래프에 대하여 검증
import matplotlib.pyplot as plt


r_X_train, v_X_train, r_y_train, v_y_train = \
train_test_split(X_train, y_train, test_size= 0.3, random_state=321)

r_X_train_prediction = pipe.predict(r_X_train)
get_evaluation_negative(r_y_train, r_X_train_prediction)

v_X_train_prediction = pipe.predict(v_X_train)
get_evaluation_negative(v_y_train, v_X_train_prediction)


fig, ax = plt.subplots(1, 2, figsize= (12, 5))

ax[0].scatter(r_y_train, r_X_train_prediction, edgecolors='red', c='red', alpha=0.2)
ax[0].plot([r_y_train.min(), r_y_train.max()], [r_y_train.min(), r_y_train.max()], 'k--')
ax[0].set_title('Train Data Prediction')

ax[1].scatter(v_y_train, v_X_train_prediction, edgecolors='red', c='blue', alpha=0.2)
ax[1].plot([v_y_train.min(), v_y_train.max()], [v_y_train.min(), v_y_train.max()], 'k--')
ax[1].set_title('Validation Data Prediction')
plt.show()

import matplotlib.pyplot as plt

prediction = pipe.predict(X_test)
get_evaluation_negative(y_test, prediction)

fig, ax = plt.subplots()
ax.scatter(y_test, prediction, edgecolors='red', c='orange', alpha=0.2)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.show()
```

<img width="670" alt="스크린샷 2024-05-15 오후 11 16 04" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/345265cd-fbf5-4b1b-af63-edb03486e2b6">
<img width="378" alt="스크린샷 2024-05-15 오후 11 16 08" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/e453e9ac-7a40-426f-8363-9e5ff58f44dc">

</details>


<h2 id='cycle03'>Cycle03</h2>
<p>1. 모델의 신뢰성을 높이기 위해 교차검증</p>

<details>
  <summary>Click Cycle03_code</summary>

```
from sklearn.model_selection import cross_val_score, KFold

features, targets = c2_h_df.iloc[:,:-1], c2_h_df.iloc[:,-1]

kf = KFold(n_splits=10, random_state=321, shuffle=True)
scores = cross_val_score( lgb_r, features, targets , cv=kf)
scores
```
<img width="360" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/33ba3740-380f-4c96-a7da-4f5557881116">

```
# 파이프라인 구축 후 차원 축소 후 선형 회귀 분석
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler


features, targets = c2_h_df.iloc[:,:-1], c2_h_df.iloc[:,-1]

X_train, X_test, y_train, y_test = \
train_test_split(features,targets, test_size=0.2, random_state=321)

kfold = KFold(n_splits=10, random_state=321, shuffle=True)


parameters = {
    # 'lgb_r__num_leaves': [10, 20, 30],
    # 'lgb_r__learning_rate': [0.05, 0.1, 0.15],
    # 'lgb_r__n_estimators': [50],
    # 'lgb_r__reg_lambda': [10000]  # L2 규제 추가
    'lgb_r__random_state': [321]
}

pipe = Pipeline(
    [
        ('pca', PCA(n_components=8)),
        ('lgb_r', LGBMRegressor())
    ]
)

grid_lgb = GridSearchCV(pipe, param_grid=parameters, cv=kfold, scoring='r2')
grid_lgb.fit(X_train, y_train)

# 최적의 파라미터와 성능 출력
print("Best parameters:", grid_lgb.best_params_)
print("Best cross-validation score: {:.3f}".format(grid_lgb.best_score_))

prediction = grid_lgb.predict(X_test)
get_evaluation_negative(y_test, prediction)

- MSE: 0.3039, RMSE: 0.5513, R2: 0.7087
```

```
import matplotlib.pyplot as plt


r_X_train, v_X_train, r_y_train, v_y_train = \
train_test_split(X_train, y_train, test_size= 0.3, random_state=321)

r_X_train_prediction = grid_lgb.predict(r_X_train)
get_evaluation_negative(r_y_train, r_X_train_prediction)

v_X_train_prediction = grid_lgb.predict(v_X_train)
get_evaluation_negative(v_y_train, v_X_train_prediction)


fig, ax = plt.subplots(1, 2, figsize= (12, 5))

ax[0].scatter(r_y_train, r_X_train_prediction, edgecolors='red', c='red', alpha=0.2)
ax[0].plot([r_y_train.min(), r_y_train.max()], [r_y_train.min(), r_y_train.max()], 'k--')
ax[0].set_title('Train Data Prediction')

ax[1].scatter(v_y_train, v_X_train_prediction, edgecolors='red', c='blue', alpha=0.2)
ax[1].plot([v_y_train.min(), v_y_train.max()], [v_y_train.min(), v_y_train.max()], 'k--')
ax[1].set_title('Validation Data Prediction')
plt.show()
```

```
# 테스트 그래프 확인
import matplotlib.pyplot as plt

prediction = grid_lgb.predict(X_test)
get_evaluation_negative(y_test, prediction)

fig, ax = plt.subplots()
ax.scatter(y_test, prediction, edgecolors='red', c='orange', alpha=0.2)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.show()
```

</details>

    
<img width="667" alt="스크린샷 2024-05-15 오후 11 17 10" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/336d165c-1a65-48f9-a570-6e0d03e20ae4">
<img width="369" alt="스크린샷 2024-05-15 오후 11 17 15" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/98016270-1533-4264-80c6-c67ab56e2079">


<details>
  <summary>Click Graph Code</summary>
    
```
bar 그래프 첨부
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 데이터 준비
data = {
    "Model_Cycle": [
        "Cycle01 Linear", "Cycle01 Poly", "Cycle01 DecisionTree",
        "Cycle01 RandomForest", "Cycle01 GradientBoosting", "Cycle01 XGB", "Cycle01 LGBM",
        "Cycle02 LGBM Train", "Cycle02 LGBM Validation", "Cycle02 LGBM Test",
        "Cycle03 LGBM Train", "Cycle03 LGBM Validation", "Cycle03 LGBM Test"
    ],
    "R2": [
        0.5210, 0.6228, 0.8324,
        0.9118, 0.8079, 0.9196, 0.9003,
        0.7131, 0.7127, 0.7071,
        0.7156, 0.7113, 0.7083
    ]
}

df = pd.DataFrame(data)

# 막대 그래프 설정
plt.figure(figsize=(12, 8))  # 그래프 크기 조절
bar_plot = sns.barplot(x="Model_Cycle", y="R2", data=df, palette="viridis")

# 사이클별 색상 지정
colors = ['red', 'green', 'blue']
cycle_colors = {cycle: colors[i % len(colors)] for i, cycle in enumerate(df['Model_Cycle'].apply(lambda x: x.split()[0]).unique())}

for bar, color in zip(bar_plot.patches, df['Model_Cycle'].apply(lambda x: cycle_colors[x.split()[0]])):
    bar.set_color(color)  # 각 막대에 색상 적용

# 각 막대에 R2 점수 텍스트 추가
for bar in bar_plot.patches:
    bar_plot.annotate(format(bar.get_height(), '.4f'),
                      (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      ha='center', va='center',
                      xytext=(0, 9),
                      textcoords='offset points')

# 레이블 및 타이틀 설정
plt.xlabel("Model and Cycle")
plt.ylabel("R2 Score")
plt.title("R2 Scores Across Different Cycles and Models")
plt.xticks(rotation=45)  # x축 레이블 회전
plt.ylim(0.4, 1.0)  # y축 범위 조정
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', axis='y', zorder=0)

# 그래프 보여주기
plt.tight_layout()
plt.show()
```

</details>

<img width="726" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/79124254-e101-4534-9d20-7b6e1b645d60">

<hr>

- 정리

    - **예측 가능한 모델 개발**: 분석을 통해 개발된 모델은 차원 축소 후에도 높은 성능을 유지하며, 병원의 퇴원 금액을 어느 정도 예측할 수 있습니다.
    - **과적합 부재**: 과적합이 관찰되지 않아, 모델에 추가적인 규제를 적용하지 않았습니다. 이는 모델이 트리 기반 회귀 방식을 사용하여 일반화된 결과를 제공하고 있음을 판단하였습니다.

- 결론

    - 본 데이터셋은 매우 높은 신뢰성을 가지고 있다고 판단되며, 환자의 퇴원 금액에 대해 다양한 요소들이 높은 상관관계를 보여 주었습니다.
    - 분석 결과, 데이터의 다양한 요소들은 높은 상관관계를 보이며, 이를 기반으로 높은 성능의 회귀 모델을 구축할 수 있었습니다.
    - 현재 모델은 주로 의사의 면허 정보에 기반하여 구축되었으나, 진료 분과 같은 추가적인 정보가 포함된다면 모델의 일반화 능력을 더욱 향상시킬 수 있을 것으로 예상됩니다.
