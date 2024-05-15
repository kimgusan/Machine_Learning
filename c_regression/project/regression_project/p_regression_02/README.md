# ② Regression02 (비선형)

## 🚴 주제: 서울시 자전거 공유 가능 수 예측.

    (1) 데이터 원본: ex) https://kaggle.com)

### 목차

1. **가설 설정**
2. **데이터 분석**
3. **데이터 전처리**
4. **데이터 훈련**
    <details>
        <summary>Cycle</summary>   
        <ul style='list-style-type: none;'>
            <li><a href="#cycle01">Cycle01(타겟형 데이터를 제외한 모델 훈련 진행)</a></li>
            <li><a href='#cycle02'>Cycle02(기존 범주형 데이터를 추가한 후 훈련 진행)</a></li>
            <li><a href='#cycle03'>Cycle03(이상치 제거 후 훈련 진행)</a></li>
            <li><a href='#cycle04'>Cycle04(과적합의 정도를 판단히기 위해 교차검증을 진행)</a></li>
            <li><a href='#cycle05'>Cycle05(다중공선성과 상관관계확인 후 불필요 featrue 제거)</a></li>
            <li><a href='#cycle06'>Cycle06(powertransform 일바화 및 L2 규제 사용하여 과적합 방지 훈련 진행)</a></li>
            <li><a href='#cycle07'>Cycle07(다중공선성과 상관관계확인 후 불필요 featrue 제거)</a></li>
        </ul>
   </details>
5. **결론**

<hr>

### 1. 가설 설정

#### 가설 1: 요소들과 자전거 배치 개수 간의 상관관계

-   **가설 내용**  
    여러 요소들이 자전거 대여 배치 수량에 영향을 미칠 수 있다고 가정합니다.  
    이 요소들과 자전거 배치 개수 간의 상관관계를 분석하여 해당 요소들이 어느 정도의 자전거 배치 모델을 관리 할 수 있다고 가정합니다.

#### 가설 2: 자전거 수량 예측을 통한 효율적 관리 가능성

-   **가설 내용**  
    예측 모델을 통해 공유할 수 있는 자전거의 수를 예측함으로써, 자전거 공유 회사가 자전거 수를 보다 효과적으로 계획하고 관리할 수 있을 것으로 기대합니다.

### 2. 데이터 분석

#### 통계적 분석 방법

-   수집된 데이터에 대해 상관관계 분석, 회귀 분석 등을 실시하여 각 요소가 자전거 배치 개수에 미치는 영향의 정도와 방향을 파악합니다.

#### 모델링과 최적화

-   분석 결과를 바탕으로 예측 모델을 생성하고, 이 모델의 일반화를 통해 최적화를 진행합니다.

<hr>

### 2. 데이터 분석

```
import pandas as pd

b_df = pd.read_csv('../../datasets/p_bike.csv')
b_df

origin_b_df = b_df.copy()
```
<hr>

### 3. 데이터 전처리

```
# 범주형 데이터를 제외한 데이터 프레임 생성
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

# 중복값 확인/
pre_l_df = pre_l_df.drop_duplicates().reset_index(drop=True)

# 상관관계 확인
pre_l_df.corr()['WORK_LIFE_BALANCE_SCORE'].sort_values(ascending=False)[1:]
```
<img width="451" alt="스크린샷 2024-05-15 오후 10 37 24" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/1c4f5394-80ed-4d86-b741-e1bd29a331af">
<img width="493" alt="스크린샷 2024-05-15 오후 10 37 32" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/499ce314-f9a1-4059-87f0-f0bf0443e7c7">



<hr>

### 4. 데이터 훈련

<h2 id="cycle01">Cycle01</h2>
<p>1. 타겟형 데이터를 제외한 모델 훈련 진행, r2 score 확인</p>

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
<img width="243" alt="스크린샷 2024-05-15 오후 10 39 21" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/cb7a8afc-a061-4bf5-85a3-b9b7c64b67d9">

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
<img width="240" alt="스크린샷 2024-05-15 오후 10 39 25" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/2ea6d90c-382e-4e93-99c6-73d1c04ad3ca">

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
```
<img width="675" alt="스크린샷 2024-05-15 오후 10 39 51" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/5e3f3d3f-083a-4df9-9175-ae07a2e7b8ad">
<img width="391" alt="스크린샷 2024-05-15 오후 10 40 17" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/4d1b89d9-d4d6-4994-b20e-7347e8739873">



```
# C01
- 비선형훈련에서 더 높은 점수를 보이고 있기 때문에 비선형 데이터 확인.
- val data 와 train 데이터에서 큰 차이는 보이지 않기 때문에 별도의 과적합인지 판단 불가
- 이상치가 존재하는 것으로 확인
```

<h2 id="cycle02">Cycle02</h2>
<p>기존 범주형 데이터를 추가한 후 훈련 진행</p>

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
```

```
# 상관관계 확인
origin_b_df.corr()['target'].sort_values(ascending=False)[1:]
```
<img width="195" alt="스크린샷 2024-05-15 오후 10 41 30" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/034ae726-9bab-4543-8efa-1566b90f1017">
<img width="500" alt="스크린샷 2024-05-15 오후 10 42 08" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/520dd237-d6db-4507-a723-14c9a6f8704c">


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

<img width="243" alt="스크린샷 2024-05-15 오후 10 43 08" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/3feb2402-e2a8-4480-adfa-3ec5851c589a">

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

<img width="245" alt="스크린샷 2024-05-15 오후 10 42 39" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/115bf70e-b11d-41cd-b6b0-4705d561c11a">

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
```

<img width="245" alt="스크린샷 2024-05-15 오후 10 42 39" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/115bf70e-b11d-41cd-b6b0-4705d561c11a">

```
# C02
범주형 데이터 추가 시 비선형 데이터 수치가 높아지는 부분 확인
트리기반의 모델에서 훈련이 더 잘되는 것을 확인하여 트리기반의 모델 선택
```

<h2 id="cycle03">Cycle03</h2>
<p>이상치를 확인하였기 때문에 이상치 제거 후 훈련 진행</p>

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
<img width="244" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/32b007c5-2b8a-46b4-bcbd-1612e3a0c6f6">

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

<img width="240" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/74b3609d-fad1-4389-86d6-a806bd6ced3d">

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

```
<img width="549" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/4dced628-6f61-4851-80a3-19516294b517">

<img width="672" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/17c5bdea-35a7-4665-b92e-1cc693132cc1">
<img width="384" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/62c2343a-c6b0-48e7-bc08-f40f41cb5e6b">


<h2 id="cycle04">Cycle04</h2>
<p>과적합의 정도를 판단히기 위해 교차검증을 진행</p>
<p>이전 검사에서 데이터의 정도를 봤을 때 과적합이 있을 수 있다는 판단</p>

```
from sklearn.model_selection import cross_val_score, KFold

features, targets = origin_b_df.iloc[:,:-1], origin_b_df.iloc[:,-1]

kf = KFold(n_splits=10, random_state=124, shuffle=True)
scores = cross_val_score(lgb_r, features, targets , cv=kf)
scores
```

<img width="356" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/3c6cfa3f-85f0-43c3-a07c-70fc6460ad30">

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
```

<img width="669" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/2a1afd3e-d843-466d-8add-09a14d6d03d0">
<img width="377" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/4c4f900e-f56b-4c8e-a7ea-5a31808a314e">




<h2 id='cycle05'>Cycle05</h2>
<p>모델의 일반화를 위해 다중공선성과 상관관계확인 후 불필요 featrue 제거.</p>

```
origin_b_df = origin_b_df.drop(labels = ['Holiday'], axis = 1)
origin_b_df = origin_b_df.drop(labels = ['Humidity(%)', 'Wind speed (m/s)', 'Solar Radiation (MJ/m2)'], axis = 1)
```

<img width="198" alt="스크린샷 2024-05-15 오후 10 51 00" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/5449dac4-43f4-4516-8100-e8e33278d2c8">
<img width="477" alt="스크린샷 2024-05-15 오후 10 51 25" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/d1dd7174-b767-4c9d-8b15-0065e57ca0e0">


```
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
```

<img width="677" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/fd40add0-21dd-4545-b6bc-c9930a04798d">
<img width="391" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/6a045345-f947-47aa-8e35-7c7680b61d83">



<h2 id='cycle06'>Cycle06</h2>
<p>각 수치형 데이터에 대하여 powertransform 을 사용하여 일반화 강화</p>
<p>Ridge 규제를 통해 과적함을 추가적으로 방지</p>

```
from sklearn.preprocessing import PowerTransformer

columns = pre_b_df.iloc[:, :-1].columns
power_b_df = pre_b_df.copy()

for column in columns:
    ptf = PowerTransformer(standardize=False)
    result = ptf.fit_transform(pre_b_df[[column]])
    power_b_df[column] = result

power_b_df


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
```

<img width="668" alt="스크린샷 2024-05-15 오후 10 54 16" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/68c45d1c-bdb7-4b96-addd-f8636b4549f1">
<img width="411" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/cc6f7c90-9e70-4001-a682-463e180d117a">


<h2 id='cycle07'>Cycle07</h2>
<p>수치폭을 더 좁힐 수 있는 지 추가 전처리 진행</p>
<p>불필요 feature 삭제.</p>

```
    power_b_df = power_b_df.drop(labels = ['Snowfall (cm)'], axis=1)

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
<img width="191" alt="스크린샷 2024-05-15 오후 10 56 11" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/1d4ec3f0-4180-487a-9b19-7b78e66669f3">
<img width="500" alt="스크린샷 2024-05-15 오후 10 56 27" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/152390bd-2e9b-4934-92b4-1f31707f47f7">

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

import matplotlib.pyplot as plt

prediction = g_lgb_r.predict(X_test)
get_evaluation_negative(y_test, prediction)

fig, ax = plt.subplots()
ax.scatter(y_test, prediction, edgecolors='red', c='orange', alpha=0.2)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.show()
```


<img width="664" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/8a4f0d24-c7f6-49fa-92bd-6a9236670b8f">
<img width="380" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/abc9c46a-f193-4d42-8b32-ff07234be844">

<img width="735" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/ad167a09-729a-4da8-8d51-d4efaf730d50">
<img width="730" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/f4b05b04-870f-4e6c-bff0-6f2b98515aff">

<hr>

-   정리

    - 과적합 문제 식별: Validation 그래프를 통해 Train 데이터와의 성능 차이를 확인했습니다. 이를 통해 모델에 과적합이 있음을 판단했습니다.
    - 모델 일반화 개선: 다중공선성 검사, Power Transform 적용, 상관관계 분석을 통해 모델의 일반화를 도모했습니다. 이러한 분석을 통해 모델이 더욱 강건해질 수 있도록 조치를 취했습니다.
    - L2 규제 적용: 교차 검증을 통해 L2 규제를 적용함으로써 과적합을 방지하였습니다. 이를 통해 검증 데이터에서의 모델 성능을 개선했습니다.

-   결론
  
    - 이 프로젝트를 통해 자전거 대여 수량에 영향을 미치는 주요 요소들을 파악하고, 이를 기반으로 효과적인 자전거 배치에 대하여 분석할 수 있는 모델을 수립할 수 있게 되었습니다.
    - 모델의 과적합 문제를 식별하고 이를 개선하는 방법을 적용함으로써, 모델의 신뢰성과 일반화 능력을 높였습니다.
    - 이 결과는 자전거 공유 회사에게 자전거 배치 계획을 더 잘 수립하고, 고객의 불편성을 해소하며 회원 유치에 도움이 될 것으로 사료됩니다.
