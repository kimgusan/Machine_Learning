# ① Regression01

## 주제: # 🏥 뉴욕주립병원 입원환자 퇴원 금액

    (1) 데이터 원본: ex) https://kaggle.com)

### 목차

1. **가설 설정**
2. **데이터 분석**
3. **데이터 전처리**
4. **데이터 훈련**
    - Cycle (n)반복 - 각 이미지에 대하여 링크를 걸어서 확인할 것.
5. **결론**

## 1. 가설 설정

### 가설 1: 의료 서비스 상관관계 분석

-   **목적**: 특정 진단 코드 또는 절차 코드에 따라 더 많은 의료 보장이 필요한 지역을 식별합니다.
-   **방법**: 다양한 지불 소스 간의 상관 관계를 분석하여, 병원이 입원 환자 방문 중 비용 효율성을 향상시킬 수 있는 자원 할당을 최적화합니다.
-   **응용**: Medicaid와 민간 보험과 같은 다양한 지불 소스 간의 패턴을 식별하여, 지역별 의료 서비스의 효율성을 높입니다.

### 가설 2: 자전거 수량 예측을 통한 효율적 관리 가능성

-   **퇴원 금액 상관관계**: 병원 퇴원 시 측정되는 금액과 다양한 요소들 간의 상관관계를 분석하여, 금액에 대한 회귀 분석을 수행합니다.
-   **응용 가능성**:
    -   **진단 코드와 의료 보장**: 필요한 의료 보장이 더 많이 필요한 지역을 진단하고, 이를 통해 의료 서비스의 효율성을 높입니다.
    -   **비용 효율성과 지불 소스 분석**: 병원은 다양한 지불 소스 간의 상관 관계 분석을 통해 비용 효율성을 향상시킬 수 있습니다.

## 2. 데이터 분석 결과

-   **예측 가능한 모델 개발**: 분석을 통해 개발된 모델은 차원 축소 후에도 높은 성능을 유지하며, 병원의 퇴원 금액을 어느 정도 예측할 수 있습니다.
-   **과적합 부재**: 과적합이 관찰되지 않아, 모델에 추가적인 규제를 적용하지 않았습니다. 이는 모델이 트리 기반 회귀 방식을 사용하여 일반화된 결과를 제공하고 있음을 의미합니다.
-   **사용된 기술**:
    -   **트리 기반 회귀 모델**: 복잡한 데이터 구조에서 유의미한 인사이트를 추출하는 데 효과적입니다.

<hr>

### 2. 데이터 분석

```

```

### 3. 데이터 전처리

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

### 4. 데이터 훈련

-   Cycle01
    1. 타겟 데이터 분포가 일정하여 차원 축소 없이 분석 진행

```
- 전체 그래프에 대해서 첨부

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

```
- OLS 지표 첨부
```

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

-   Cycle02
    1. 모델의 공분산성을 지닌 수치 중 높은 수치에 대하여 삭제 후 확인
    2. 모델 훈련속도의 효율을 높이기 위해 차원축소를 진행.

```
# 상관관계 확인
num_h_df.corr()['Total Charges'].sort_values(ascending=False)[1:]

import seaborn as sns
corr = num_h_df.corr()
sns.heatmap(corr, cmap='Oranges')
- 히트맵그래프 첨부

```

```
from statsmodels.stats.outliers_influence import variance_inflation_factor

def get_vif(features):
    vif = pd.DataFrame()
    vif['vif_score'] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
    vif['feature'] = features.columns
    return vif

get_vif(features)

- 다중공선성 수치 그래프 첨부
```

```
# 불필요 feature 제거
c2_h_df = num_h_df.drop(labels = ['Operating Certificate Number', 'APR DRG Code', 'Health Service Area'], axis = 1)
```

```
# 차원축소 진행
from sklearn.model_selection import train_test_split

features, targets = c2_h_df.iloc[:, :-1], c2_h_df.iloc[:, -1]

X_train, X_test, y_train, y_test = \
train_test_split(features, targets, test_size=0.2, random_state=124)


# 손실율 확인
from sklearn.decomposition import PCA

for i in range(4):
    pca = PCA(n_components=(i + 1))

    pca_train = pca.fit_transform(X_train)

    # 손실율
    print(pca.explained_variance_ratio_.sum())
```

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

- 해당 결과값에 대하여 복사해서 붙여넣을 것.
```

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

- 그래프 첨부

import matplotlib.pyplot as plt

prediction = pipe.predict(X_test)
get_evaluation_negative(y_test, prediction)

fig, ax = plt.subplots()
ax.scatter(y_test, prediction, edgecolors='red', c='orange', alpha=0.2)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.show()

- 테스트 그래프 첨부
```

-   Cycle03
    1. 모델의 신뢰성을 높이기 위해 교차검증

```
from sklearn.model_selection import cross_val_score, KFold

features, targets = c2_h_df.iloc[:,:-1], c2_h_df.iloc[:,-1]

kf = KFold(n_splits=10, random_state=321, shuffle=True)
scores = cross_val_score( lgb_r, features, targets , cv=kf)
scores

[0.89412771, 0.89391949, 0.89559464, 0.89305629, 0.89536895,
       0.89599536, 0.89029503, 0.89492653, 0.89381586, 0.89115958])
```

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
# trian, validation 그래프 확인
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

- 그래프 파임 첨부 예정
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

- 그래프 파일 첨부 예정
```

```
bar 그래프 첨부 진행
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

-   결과

    분석 결과:

    예측 가능한 모델의 개발: 분석을 통해 개발된 모델은 여러 요소들을 고려하여 병원의 퇴원 금액을 어느 정도 예측할 수 있음을 보여줍니다. 모델은 차원 축소 후에도 높은 성능을 유지하고 있습니다. 과적합의 부재: 과적합이 관찰되지 않아 모델에 추가적인 규제를 적용하지 않았습니다. 이는 모델이 훈련 데이터에 대해 지나치게 최적화되지 않고 일반화된 결과를 제공하고 있음을 의미합니다. 사용된 기술:

    트리 기반 회귀 모델: 트리 기반의 회귀 모델을 사용하여 분석을 수행하였으며, 이 모델은 복잡한 데이터 구조에서 유의미한 인사이트를 추출하는 데 효과적이었
