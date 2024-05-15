# ① Regression01

## 주제: 😊 직업적 삶과 개인 생활 전반적인 만족 점수

    (1) 데이터 원본: ex) https://kaggle.com)

### 목차

1. **가설 설정**
2. **데이터 분석**
3. **데이터 전처리**
4. **데이터 훈련**
    - Cycle (n)반복 - 각 이미지에 대하여 링크를 걸어서 확인할 것.
5. **결론**

### 1. 가설 설정

#### 가설 1: 요소들과 자전거 배치 개수 간의 상관관계

-   **가설 내용**  
    여러 요소들이 자전거 대여 수량에 영향을 미칠 수 있다고 가정합니다.  
    이 요소들과 자전거 배치 개수 간의 상관관계를 분석하여 어떤 요소들이 가장 큰 영향을 미치는지 파악합니다.

#### 가설 2: 자전거 수량 예측을 통한 효율적 관리 가능성

-   **가설 내용**  
    예측 모델을 통해 공유할 수 있는 자전거의 수를 예측함으로써, 자전거 공유 회사가 자전거 수를 보다 효과적으로 계획하고 관리할 수 있을 것으로 기대합니다.

### 2. 데이터 분석

#### 통계적 분석 방법

-   수집된 데이터에 대해 상관관계 분석, 회귀 분석 등을 실시하여 각 요소가 자전거 배치 개수에 미치는 영향의 정도와 방향을 파악합니다.

#### 모델링과 최적화

-   분석 결과를 바탕으로 예측 모델을 생성하고, 이 모델을 최적화하여 가설의 타당성을 평가합니다

<hr>

### 2. 데이터 분석

```
# 범주형 데이터를 제외한 데이터 프레임 생성
origin_b_df = b_df.copy()
pre_b_df = b_df.copy()
pre_b_df = pre_b_df.drop(labels = ['Date', 'Seasons','Holiday','Functioning Day'], axis=1)

# 타겟 컬럼 위치 변경
target_column =  pre_b_df.pop('Rented Bike Count')
pre_b_df.loc[:, 'target'] = target_column
pre_b_df

pre_b_df.isna().sum()
pre_b_df.duplicated().sum()

# 상관관계 확인
pre_b_df.corr()['target'].sort_values(ascending=False)[1:]

- 시각화 이미지 첨부

- OLS 지표 첨부


```

### 3. 데이터 전처리

```
# 중복값 확인/
pre_l_df = pre_l_df.drop_duplicates().reset_index(drop=True)

# 상관관계 확인
pre_l_df.corr()['WORK_LIFE_BALANCE_SCORE'].sort_values(ascending=False)[1:]

- 이미지 회귀선 넣기
- corr 이미지 넣기
- 상관관계 이미지 넗기
- 선형 이미지 넣기
- 양의 상관관계 음의 상관관계 넣기
- 다중공산성 지표 넣기
- OLS 지표 넣기

```

### 4. 데이터 훈련

-   Cycle01
    1. 타겟형 데이터를 제외한 모델 훈련 진행, r2 score 확인

```
# 선형 데이터 확인
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

features, targets = pre_b_df.iloc[:, :-1], pre_b_df.iloc[:, -1]

X_train, X_test, y_train, y_test = \
train_test_split(features, targets, test_size=0.2, random_state=321)

l_r = LinearRegression()
l_r.fit(X_train, y_train)

prediction = l_r.predict(X_test)
get_evaluation_negative(y_test, prediction)

MSE: 208294.7827, RMSE: 456.3932, R2: 0.4893
```

```
# 비선형 데이터 확인
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_features = PolynomialFeatures(degree=3).fit_transform(features)

X_train, X_test, y_train, y_test =\
train_test_split(poly_features, targets, test_size=0.2, random_state=321)

l_r = LinearRegression()
l_r.fit(X_train, y_train)

prediction = l_r.predict(X_test)
get_evaluation_negative(y_test, prediction)

MSE: 140395.6319, RMSE: 374.6941, R2: 0.6558
```

```
# 훈련 데이터와 검증 데이터 그래프 확인
import matplotlib.pyplot as plt


r_X_train, v_X_train, r_y_train, v_y_train = \
train_test_split(X_train, y_train, test_size= 0.3, random_state=321)

r_X_train_prediction = l_r.predict(r_X_train)
get_evaluation_negative(r_y_train, r_X_train_prediction)

v_X_train_prediction = l_r.predict(v_X_train)
get_evaluation_negative(v_y_train, v_X_train_prediction)


fig, ax = plt.subplots(1, 2, figsize= (12, 5))

ax[0].scatter(r_y_train, r_X_train_prediction, edgecolors='red', c='red', alpha=0.2)
ax[0].plot([r_y_train.min(), r_y_train.max()], [r_y_train.min(), r_y_train.max()], 'k--')
ax[0].set_title('Train Data Prediction')

ax[1].scatter(v_y_train, v_X_train_prediction, edgecolors='red', c='blue', alpha=0.2)
ax[1].plot([v_y_train.min(), v_y_train.max()], [v_y_train.min(), v_y_train.max()], 'k--')
ax[1].set_title('Validation Data Prediction')
plt.show()


    MSE: 128546.7844, RMSE: 358.5342, R2: 0.6928
    MSE: 132403.7434, RMSE: 363.8733, R2: 0.6823

- 훈련 데이터 검증 그래프 첨부

```

```
- test 검사 함수.

import matplotlib.pyplot as plt

prediction = l_r.predict(X_test)
get_evaluation_negative(y_test, prediction)

fig, ax = plt.subplots()
ax.scatter(y_test, prediction, edgecolors='red', c='orange', alpha=0.2)
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--')
plt.show()
- test 검사 첨부
```

```
# C01
- 비선형훈련에서 더 높은 점수를 보이고 있기 때문에 비선형 데이터 확인.
- val data 와 train 데이터에서 큰 차이는 보이지 않기 때문에 별도의 과적합인지 판단 불가
- 이상치가 존재하는 것으로 확인
```

```
    # 훈련 결과 Cycle_01
    prediction = l_r.predict(X_test)
    get_evaluation(y_test, prediction)
```

-   Cycle02
    -   기존 범주형 데이터를 추가한 후 훈련 진행

```
# 범주형 데이터 레이블인코딩
from sklearn.preprocessing import LabelEncoder

columns = ['Seasons', 'Holiday']
label_encoders = {}

for column in columns:
    encoder = LabelEncoder()
    result = encoder.fit_transform(origin_b_df[column])
    origin_b_df[column] = result
    label_encoders[column] = encoder.classes_

label_encoders

- OLS 지표 첨부
```

```
# Lineare Regresssion 훈련
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

features, targets = origin_b_df.iloc[:, :-1], origin_b_df.iloc[:, -1]

X_train, X_test, y_train, y_test = \
train_test_split(features, targets, test_size=0.2, random_state=321)

l_r = LinearRegression()
l_r.fit(X_train, y_train)

prediction = l_r.predict(X_test)
get_evaluation_negative(y_test, prediction)

MSE: 203459.3725, RMSE: 451.0647, R2: 0.5012

```

```
# 비선형모델 훈련
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_features = PolynomialFeatures(degree=2).fit_transform(features)

X_train, X_test, y_train, y_test =\
train_test_split(poly_features, targets, test_size=0.2, random_state=321)

l_r = LinearRegression()
l_r.fit(X_train, y_train)

prediction = l_r.predict(X_test)
get_evaluation_negative(y_test, prediction)

MSE: 162936.3859, RMSE: 403.6538, R2: 0.6005
```

```
# 트리 모델 훈련
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

features, targets = origin_b_df.iloc[:, :-1], origin_b_df.iloc[:, -1]

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
    get_evaluation_negative(y_test, prediction)

DecisionTreeRegressor
MSE: 180518.2426, RMSE: 424.8744, R2: 0.5574
RandomForestRegressor
MSE: 85586.0729, RMSE: 292.5510, R2: 0.7902
GradientBoostingRegressor
MSE: 91201.5446, RMSE: 301.9959, R2: 0.7764
XGBRegressor
MSE: 86721.8676, RMSE: 294.4858, R2: 0.7874
LGBMRegressor
MSE: 81825.2797, RMSE: 286.0512, R2: 0.7994
```

```
범주형 데이터 추가 시 비선형 데이터 수치가 높아지는 부분 확인
트리기반의 모델에서 훈련이 더 잘되는 것을 확인하여 트리기반의 모델 선택
```

-   Cycle03
    -   이상치를 확인하였기 때문에 이상치 제거 후 훈련 진행

```
# 이상치 제거를 위한 표준화 작업
from sklearn.preprocessing import StandardScaler

std = StandardScaler()
result = std.fit_transform(origin_b_df)
std_origin_b_df = pd.DataFrame(result, columns = origin_b_df.columns)
std_origin_b_df

# 이상치 확인 및 제거
condition = True
error_count = []

for column in std_origin_b_df.columns:
    if std_origin_b_df[column].between(-1.96, 1.96) is True:
        error_count.append(std_origin_b_df[column].between(-1.96, 1.96).count())
    condition &= std_origin_b_df[column].between(-1.96, 1.96)

std_origin_b_df = std_origin_b_df[condition]
std_origin_b_df

# 이상치 제거한 데이터를 인덱스 번호에 맞게 가져오기
origin_b_df = origin_b_df.iloc[std_origin_b_df.index].reset_index(drop=True)
origin_b_df
```

```
# 선형 회귀 모델 훈련
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

features, targets = origin_b_df.iloc[:, :-1], origin_b_df.iloc[:, -1]

X_train, X_test, y_train, y_test = \
train_test_split(features, targets, test_size=0.2, random_state=321)

l_r = LinearRegression()
l_r.fit(X_train, y_train)

prediction = l_r.predict(X_test)
get_evaluation_negative(y_test, prediction)

MSE: 141044.1497, RMSE: 375.5585, R2: 0.4673
```

```
# 비선형 모델 확인
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_features = PolynomialFeatures(degree=2).fit_transform(features)

X_train, X_test, y_train, y_test =\
train_test_split(poly_features, targets, test_size=0.2, random_state=321)

l_r = LinearRegression()
l_r.fit(X_train, y_train)

prediction = l_r.predict(X_test)
get_evaluation_negative(y_test, prediction)

MSE: 125052.8676, RMSE: 353.6281, R2: 0.5277
```

```
# 트리 모델 확인
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

features, targets = origin_b_df.iloc[:, :-1], origin_b_df.iloc[:, -1]

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
    get_evaluation_negative(y_test, prediction)

DecisionTreeRegressor
MSE: 124514.8276, RMSE: 352.8666, R2: 0.5297
RandomForestRegressor
MSE: 75376.0106, RMSE: 274.5469, R2: 0.7153
GradientBoostingRegressor
MSE: 83471.2015, RMSE: 288.9138, R2: 0.6847
XGBRegressor
MSE: 81891.1387, RMSE: 286.1663, R2: 0.6907
LGBMRegressor
MSE: 76253.2006, RMSE: 276.1398, R2: 0.7120
```

```
- 트리 모델에 대현 validation train 이미지 첨부
- test 데이터에 대한 이미지 첨부
```

-   Cycle04
    -   과적합의 정도를 판단히기 위해 교차검증을 진행
    -   이전 검사에서 데이터의 정도를 봤을 때 과적합이 있을 수 있다는 판다.

```
from sklearn.model_selection import cross_val_score, KFold

features, targets = origin_b_df.iloc[:,:-1], origin_b_df.iloc[:,-1]

kf = KFold(n_splits=10, random_state=124, shuffle=True)
scores = cross_val_score(lgb_r, features, targets , cv=kf)
scores

- score에 대한 이미지 첨부하기
```

```
# 교차검증 및 l2 규제
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline


features, targets = origin_b_df.iloc[:,:-1], origin_b_df.iloc[:,-1]

X_train, X_test, y_train, y_test = \
train_test_split(features,targets, test_size=0.2, random_state=321)

kfold = KFold(n_splits=10, random_state=124, shuffle=True)

lgb_r = LGBMRegressor()

parameters = {
    'random_state': [321],
    # 'min_gain_to_split' : [0.01],
    # 'max_depth' : [10],
    # 'num_leaves' : [31],
    # 'reg_lambda': [100],
    'verbose': [-1]
}

g_lgb_r = GridSearchCV(lgb_r, param_grid=parameters, cv=kfold, scoring='neg_mean_squared_error')
g_lgb_r.fit(X_train, y_train)

# 최적의 파라미터와 성능 출력
print("Best parameters:", g_lgb_r.best_params_)
print("Best cross-validation score: {:.3f}".format(-g_lgb_r.best_score_))

prediction = g_lgb_r.predict(X_test)
get_evaluation_negative(y_test, prediction)

MSE: 76253.2006, RMSE: 276.1398, R2: 0.7120

- train 하고 validation그래프 첨부하기
- test 데이터 처뭅하기
```

-   Cycle05
    -   모델의 일반화를 위해 다중공산성과 상관관계확인ㄹ후 불필요 featrue 제거.

```
- OLS지표 확인
- 상관관계 확인
origin_b_df = origin_b_df.drop(labels = ['Holiday'], axis = 1)
- 데이터 전처리 후 OLS 지표 확인
origin_b_df = origin_b_df.drop(labels = ['Humidity(%)', 'Wind speed (m/s)', 'Solar Radiation (MJ/m2)'], axis = 1)
- 데이터 전처리 후 OLS 지표 확인


from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline


features, targets = origin_b_df.iloc[:,:-1], origin_b_df.iloc[:,-1]

X_train, X_test, y_train, y_test = \
train_test_split(features,targets, test_size=0.2, random_state=321)

kfold = KFold(n_splits=10, random_state=124, shuffle=True)

lgb_r = LGBMRegressor()

parameters = {
    'random_state': [321],
    # 'reg_lambda': [100],
    'verbose': [-1]
}

g_lgb_r = GridSearchCV(lgb_r, param_grid=parameters, cv=kfold, scoring='neg_mean_squared_error')
g_lgb_r.fit(X_train, y_train)

# 최적의 파라미터와 성능 출력
print("Best parameters:", g_lgb_r.best_params_)
print("Best cross-validation score: {:.3f}".format(-g_lgb_r.best_score_))

prediction = g_lgb_r.predict(X_test)
get_evaluation_negative(y_test, prediction)


MSE: 73783.6514, RMSE: 271.6315, R2: 0.7213

- train, validation data 그래프 확인
- test 데이터 확인
```

-   Cycle06
    -   각 수치형 데이터에 대하여 powertransform 을 사용하여 일반화 강화
    -   Ridge 규제를 통해 과적함을 추가적으로 방지

```
from sklearn.preprocessing import PowerTransformer

columns = pre_b_df.iloc[:, :-1].columns
power_b_df = pre_b_df.copy()

for column in columns:
    ptf = PowerTransformer(standardize=False)
    result = ptf.fit_transform(pre_b_df[[column]])
    power_b_df[column] = result

power_b_df

- OLS 지표 첨부

- 데이터 전처리
power_b_df = power_b_df.drop(labels = ['Wind speed (m/s)'], axis = 1)

from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline


features, targets = power_b_df.iloc[:,:-1], power_b_df.iloc[:,-1]

X_train, X_test, y_train, y_test = \
train_test_split(features,targets, test_size=0.2, random_state=321)

kfold = KFold(n_splits=10, random_state=124, shuffle=True)

lgb_r = LGBMRegressor()

parameters = {
    'random_state': [321],
    'reg_lambda': [100],
    'verbose': [-1]
}

g_lgb_r = GridSearchCV(lgb_r, param_grid=parameters, cv=kfold, scoring='neg_mean_squared_error')
g_lgb_r.fit(X_train, y_train)

# 최적의 파라미터와 성능 출력
print("Best parameters:", g_lgb_r.best_params_)
print("Best cross-validation score: {:.3f}".format(-g_lgb_r.best_score_))

prediction = g_lgb_r.predict(X_test)
get_evaluation_negative(y_test, prediction)


MSE: 88352.2831, RMSE: 297.2411, R2: 0.7834


- train, validation 그래프 처부
- test 데이터 그래프 첨부
```

-   Cycle07
    -   수치폭을 더 좁힐 수 있는 지 추가 전처리 진행
    -   불필요 feature 삭제.

```
    - 상관관계 수치 그래프 확인
    - OLS 지표 확인
    - 다중공선성 지표 확인

    power_b_df = power_b_df.drop(labels = ['Snowfall (cm)'], axis=1)
    - OLS 지표 확인

    from lightgbm import LGBMRegressor
    from sklearn.model_selection import train_test_split, GridSearchCV, KFold
    from sklearn.pipeline import Pipeline


    features, targets = power_b_df.iloc[:,:-1], power_b_df.iloc[:,-1]

    X_train, X_test, y_train, y_test = \
    train_test_split(features,targets, test_size=0.2, random_state=321)

    kfold = KFold(n_splits=10, random_state=124, shuffle=True)

    lgb_r = LGBMRegressor()

    parameters = {
        'random_state': [321],
        'reg_lambda': [100],
        'verbose': [-1]
    }

    g_lgb_r = GridSearchCV(lgb_r, param_grid=parameters, cv=kfold, scoring='neg_mean_squared_error')
    g_lgb_r.fit(X_train, y_train)

    # 최적의 파라미터와 성능 출력
    print("Best parameters:", g_lgb_r.best_params_)
    print("Best cross-validation score: {:.3f}".format(-g_lgb_r.best_score_))

    prediction = g_lgb_r.predict(X_test)
    get_evaluation_negative(y_test, prediction)

    MSE: 88007.3417, RMSE: 296.6603, R2: 0.7842




```

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

X_train, X_test, y_train, y_test =\
train_test_split(features, targets, test_size=0.2, random_state=321)

r_X_train, v_X_train, r_y_train, v_y_train = \
train_test_split(X_train, y_train, test_size= 0.3, random_state=321)

# 모델 학습 및 예측

g_lgb_r.fit(r_X_train, r_y_train)
prediction_r_train = g_lgb_r.predict(r_X_train)
prediction_v_train = g_lgb_r.predict(v_X_train)

# 평가

get_evaluation_negative(r_y_train, prediction_r_train)
get_evaluation_negative(v_y_train, prediction_v_train)

# 시각화

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(r_y_train, prediction_r_train, edgecolors='red', c='red', alpha=0.2)
ax[0].plot([r_y_train.min(), r_y_train.max()], [r_y_train.min(), r_y_train.max()], 'k--')
ax[0].set_title('Train Data Prediction')

ax[1].scatter(v_y_train, prediction_v_train, edgecolors='red', c='blue', alpha=0.2)
ax[1].plot([v_y_train.min(), v_y_train.max()], [v_y_train.min(), v_y_train.max()], 'k--')
ax[1].set_title('Validation Data Prediction')
plt.show()

MSE: 69007.0353, RMSE: 262.6919, R2: 0.8351
MSE: 99120.6772, RMSE: 314.8344, R2: 0.7621

-   train, validation 그래프 첨부

import matplotlib.pyplot as plt

prediction = g_lgb_r.predict(X_test)
get_evaluation_negative(y_test, prediction)

fig, ax = plt.subplots()
ax.scatter(y_test, prediction, edgecolors='red', c='orange', alpha=0.2)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.show()
최종 각 R2 Score에 대한 막대그래프 점 첨부 필요.

```
- 시각화 그래프 사용하여 논리적으로 설명
- 시각화 그래프를 팀원중 본 사람이 없어 피트백 불가
```

-   결과

        - 가설

    목적: 자전거 대여 서비스에서 자전거 배치 개수에 영향을 미치는 주요 요소들을 식별하고 이해하기 위해 데이터를 분석합니다.

-   주요 내용:  
    여러 요소들이 자전거 대여 수량에 영향을 미칠 수 있다고 가정합니다.
    이 요소들과 자전거 배치 개수 간의 상관관계를 분석하여 어떤 요소들이 가장 큰 영향을 미치는지 파악합니다.
    예측 모델을 통해 공유할 수 있는 자전거의 수를 예측함으로써, 자전거 공유 회사가 자전거 수를 보다 효과적으로 계획하고 관리할 수 있을 것으로 기대합니다.

-   결론

-   과적합 문제 식별: Validation 그래프를 통해 Train 데이터와의 성능 차이를 확인했습니다. 이를 통해 모델에 과적합이 있음을 판단했습니다.
-   모델 일반화 개선: 다중공선성 검사, Power Transform 적용, 상관관계 분석을 통해 모델의 일반화를 도모했습니다. 이러한 분석을 통해 모델이 더욱 강건해질 수 있도록 조치를 취했습니다.
    L2 규제 적용: 교차 검증을 통해 L2 규제를 적용함으로써 과적합을 방지하였습니다. 이를 통해 검증 데이터에서의 모델 성능을 개선했습니다.

-   종합적인 관점
    이 프로젝트를 통해 자전거 대여 수량에 영향을 미치는 주요 요소들을 파악하고, 이를 기반으로 효과적인 자전거 배치 전략을 수립할 수 있게 되었습니다. 모델의 과적합 문제를 식별하고 이를 개선하는 방법을 적용함으로써, 모델의 신뢰성과 일반화 능력을 높였습니다. 이 결과는 자전거 공유 회사에게 자전거 배치 계획을 더 잘 수립하고, 고객 수요를 보다 효과적으로 충족시킬 수 있는 방안을 제공합니다.
