# ① Regression01

## 주제: 😊 직업적 삶과 개인 생활 전반적인 만족 점수

    (1) 데이터 원본: ex) https://kaggle.com)

### 목차

1. **가설 설정**
2. **데이터 분석**
3. **데이터 전처리**
4. **데이터 훈련**
    <details>
        <summary>Cycle</summary>   
        <ul style='list-style-type: none;'>
            <li><a href="#cycle01">Cycle01(선형 모델 훈련.)</a></li>
            <li><a href='#cycle02'>Cycle02(차원축소 및 선형 모델 훈련)</a></li>
            <li><a href='#cycle03'>Cycle03(과적합 판단을 위한 교차검증 진행.)</a></li>
            <li><a href='#cycle04'>Cycle04(파이토치를 사용한 최적의 loss 값 확인.)</a></li>
            <li><a href='#cycle05'>Cycle05(차원 축소 진행 후 L2 규제를 사용한 회귀 모델 분석.)</a></li>
        </ul>
   </details>
   
6. **결론**

<hr>

### 1. 가설 설정

#### 가설 1: 라이프스타일 요소와 삶의 만족도 점수 간의 상관관계

-   **가설 내용**  
    특정 라이프스타일 요소(예: 신체 활동 수준, 사회적 활동, 일과 생활의 균형 등)는 삶의 만족도 점수와 강한 상관관계를 보일 것으로 예상됩니다.
    이 가설은 라이프스타일이 개인의 일반적인 웰빙과 직접적으로 연결되어 있다는 가정으로 진행됩니다.

#### 가설 2: 긍정적 요소들이 삶의 만족도에 더 큰 영향을 미친다

-   **가설 내용**  
    긍정적 요소(예: 자기계발, 목표 달성, 사회적 지원)는 부정적 요소(예: 스트레스, 휴가 제한)보다 삶의 만족도에 더 큰 긍정적 영향을 미칠 것이다.  
    이 가설은 긍정적인 심리적, 사회적 자원이 부정적인 영향을 상쇄하고, 전반적인 만족도를 높인다는 이론에서 발생됩니다.

#### 가설 검증 방법

1. **통계적 분석**: 수집된 데이터에 대해 상관관계 분석, 회귀 분석 등을 실시하여 각 요소가 만족도에 미치는 영향의 정도와 방향을 파악합니다.

2. **모델링과 최적화**: 분석 결과를 바탕으로 예측 모델을 생성하고, 이 모델을 최적화하여 가설의 타당성을 평가합니다.  
   이러한 과정을 통해 설정된 가설의 타당성을 검증하고, 일과 삶의 만족도를 높이는 데 중요한 요소를 식별할 수 있습니다.  
   이 결과는 개인의 웰빙 점수를 개선하는데 전략 수립에 대한 중요한 부분을 확인할 수 있습니다.

<hr>

### 2. 데이터 분석

```
# 데이터 셋 정보 확인.
pre_l_df = l_df.copy()
pre_l_df.info()

# 결측치 확인.
pre_l_df.isna().sum()

# daliy stress 이상값 삭제.
pre_l_df = pre_l_df[pre_l_df['DAILY_STRESS'] != '1/1/00']
pre_l_df = pre_l_df.reset_index(drop=True)
pre_l_df

# 데이탸 셋 정보 확인.
pre_l_df.info()

```
<hr>

### 3. 데이터 전처리

```
# 중복값 확인/
pre_l_df = pre_l_df.drop_duplicates().reset_index(drop=True)

# 상관관계 확인
pre_l_df.corr()['WORK_LIFE_BALANCE_SCORE'].sort_values(ascending=False)[1:]
```

<img width="361" alt="스크린샷 2024-05-15 오후 4 45 18" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/62d1c572-2027-4233-8cff-c3ca01e49c4f">
<img width="483" alt="스크린샷 2024-05-15 오후 4 48 52" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/2ba37e31-ec58-4577-9d4d-e967b593010c">
<img width="595" alt="스크린샷 2024-05-15 오후 4 49 00" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/aa2b9580-05a7-48af-8c0a-b257b2938f92">
<img width="188" alt="스크린샷 2024-05-15 오후 4 49 26" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/781ddef3-aab6-423a-b099-32f54483e279">
<img width="525" alt="스크린샷 2024-05-15 오후 4 49 16" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/d7747240-e778-4df8-8ae0-19d4bbda1fbc">

<hr>

### 4. 데이터 훈련

<h2 id="cycle01">Cycle01</h2>
<p>1. 선형 모델 훈련</p>

```
 선형 데이터 훈련
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

features, targets = pre_l_df.iloc[:, :-1], pre_l_df.iloc[:, -1]

X_train, X_test, y_train, y_test = \
train_test_split(features, targets, test_size=0.2, random_state=124)

l_r = LinearRegression()
l_r.fit(X_train, y_train)
```

```
# 훈련 데이터 평가 함수.
import numpy as np
from sklearn.metrics import mean_squared_log_error, mean_squared_error, r2_score

def get_evaluation(y_test, prediction):
    MSE = mean_squared_error(y_test, prediction)
    RMSE = np.sqrt(MSE)
    MSLE = mean_squared_log_error(y_test, prediction)
    RMSLE = np.sqrt(mean_squared_log_error(y_test, prediction))
    R2 = r2_score(y_test, prediction)
    print('MSE: {:.4f}, RMSE: {:.4f}, MSLE: {:.4f}, RMSLE: {:.4f}, R2: {:.4f}'\
          .format(MSE, RMSE, MSLE, RMSLE, R2))
```

<img width="365" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/c267e9d4-a8b0-4b44-bfe9-06a09faee1cf">

<h2 id="cycle02">Cycle02</h2>
<p>1. 차원 축소 진행 (PCA)</p>
<p>2. 선형 데이터 훈련</p>

```
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


for i in range(12):
    pca = PCA(n_components=(i+1))

    pca_train = pca.fit_transform(features)

    print(pca.explained_variance_ratio_.sum(), i)
```
<img width="123" alt="스크린샷 2024-05-15 오후 4 52 43" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/d8241fce-2a8d-4b61-9aa9-c45ec7d8e1ea">

```
# 보존률
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())

0.9106306179404671
```

```
# 파이프라인 구축 후 차원 축소 후 선형 회귀 분석
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

features, targets = pre_l_df.iloc[:,:-1], pre_l_df.iloc[:,-1]

X_train, X_test, y_train, y_test = \
train_test_split(features,targets, test_size=0.2, random_state=321)

pipe = Pipeline(
    [
        ('pca', PCA(n_components=8)),
        ('l_r',LinearRegression())
    ]
)

pipe.fit(X_train, y_train)
```
<img width="125" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/ebd7d281-0183-47d3-8c88-5e27fa937bbd">

```
prediction = pipe.predict(X_test)
get_evaluation(y_test, prediction)

```
<img width="374" alt="스크린샷 2024-05-15 오후 4 53 36" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/39d47de8-c62c-436d-8f07-28791daeb63a">


<h2 id="cycle03">Cycle03</h2>
<p>1. 과적합을 판단하기 위해 교차검증 진행</p>
 
```
from sklearn.model_selection import cross_val_score, KFold

features, targets = pre_l_df.iloc[:,:-1], pre_l_df.iloc[:,-1]

kf = KFold(n_splits=10, random_state=321, shuffle=True)
scores = cross_val_score(pipe, features, targets , cv=kf)
scores
>
array([0.89424997, 0.88934348, 0.89652179, 0.88739977, 0.88346597,
       0.88784766, 0.8909993 , 0.88743523, 0.88928659, 0.88422674])
```

```
# 파이프라인 구축 후 차원 축소 후 선형 회귀 분석
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline


features, targets = pre_l_df.iloc[:,:-1], pre_l_df.iloc[:,-1]

X_train, X_test, y_train, y_test = \
train_test_split(features,targets, test_size=0.2, random_state=321)

kfold = KFold(n_splits=15, random_state=321, shuffle=True)

pipe = Pipeline(
    [
        ('pca', PCA(n_components=8)),  # PCA로 차원 축소
        ('l_r', LinearRegression())    # 선형 회귀 모델
    ]
)

param_grid = {
    'pca__n_components': [8]  # PCA 컴포넌트 수 조정
}

grid_l = GridSearchCV(pipe, param_grid=param_grid, cv=kfold, scoring='neg_mean_squared_error')
grid_l.fit(X_train, y_train)

# 최적의 파라미터와 성능 출력
print("Best parameters:", grid_l.best_params_)
print("Best cross-validation score: {:.3f}".format(-grid_l.best_score_))

prediction = grid_l.predict(X_test)
get_evaluation(y_test, prediction)

MSE: 221.7647, RMSE: 14.8918, MSLE: 0.0005, RMSLE: 0.0227, R2: 0.8917
```
<img width="125" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/4f0f7665-5578-4faf-ad32-0b027b0b0db4">




```
import matplotlib.pyplot as plt

prediction = grid_l.predict(X_test)
get_evaluation(y_test, prediction)

fig, ax = plt.subplots()
ax.scatter(y_test, prediction, edgecolors='red', c='orange', alpha=0.2)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.show()

```

<img width="663" alt="스크린샷 2024-05-15 오후 4 55 10" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/1b7e1c78-07b9-459b-bfec-a6347f0111c2">
<img width="373" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/13f7a7fa-2679-4848-b732-0ba19d2ed7ea">


-   Cycle04
    -   파이토치를 사용하여 loss 값을 확인하여 그래프를 확인하여 실제 train 의 값과 test 의 값이 오차가 커지는 지점을 찾아서, 규제를 확인한다.

```
import torch
from torch.optim import SGD
from torch.nn.functional import mse_loss
from torch.nn import Linear

torch.manual_seed(321)

features, targets = pre_l_df.iloc[:,:-1], pre_l_df.iloc[:,-1]

X_train, X_test, y_train, y_test = \
train_test_split(features,targets, test_size=0.2, random_state=321)

real_X_train, val_X_test, real_y_train, val_y_test = \
train_test_split(X_train, y_train, test_size=0.5, random_state=321)

real_X_train = torch.FloatTensor(X_train.values)
real_y_train = torch.FloatTensor(y_train.values).view(-1, 1)

val_X_test = torch.FloatTensor(X_test.values)
val_y_test = torch.FloatTensor(y_test.values).view(-1, 1)
l_r = Linear(real_X_train.shape[1], 1)  # 입력 차원 동적 할당
optimizer = SGD(l_r.parameters(), lr=0.001)  # 보다 실용적인 학습률

epochs = 1000000  # 적절한 에포크 수
real_train_loss_history = []
val_test_loss_history = []

for epoch in range(1, epochs + 1):
    l_r.train()
    H = l_r(real_X_train)
    train_loss = mse_loss(real_y_train, H)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    l_r.eval()
    with torch.no_grad():
        H_test = l_r(val_X_test)
        test_loss = mse_loss(val_y_test, H_test)

    real_train_loss_history.append(train_loss.item())
    val_test_loss_history.append(test_loss.item())

    # if epoch % 10000 == 0:
    #     print(f'{epoch}/{epochs}: ', end='')
    #     W = l_r.weight.data.squeeze()
    #     b = l_r.bias.data
    #     for i, w in enumerate(W):
    #         print(f'W{i+1}: {w:.4f}, ', end='')
    #     print(f'b: {b.item():.4f}, Loss: {train_loss.item():.4f}\n')
    if epoch % 10000 == 0:
        print(f'{epoch}/{epochs}: ', end='')
        W = l_r.weight.data.squeeze()
        b = l_r.bias.data
        for i, w in enumerate(W):
            print(f'W{i+1}: {w:.4f}, ', end='')
        print(f'b: {b.item():.4f}, Loss: {test_loss.item():.4f}\n')

plt.figure(figsize=(10, 5))
plt.plot(real_train_loss_history, label='Training Loss', linewidth=3, color= 'skyblue')
plt.plot(val_test_loss_history, label='Val Loss', linestyle='--' , linewidth=3, alpha=0.3, color='red')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Loss Trend Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
```

<img width="603" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/26ff1f4c-7288-47fa-803d-5588825eb39c">


-   Cycle05
    -   파이프라인을 구축하여 차원축소 진행 후 L2 규제 사용된 선형 회귀 분석 진행.

```
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

features, targets = pre_l_df.iloc[:,:-1], pre_l_df.iloc[:,-1]

X_train, X_test, y_train, y_test = \
train_test_split(features,targets, test_size=0.2, random_state=321)

pipe = Pipeline(
    [
        ('pca', PCA(n_components=8)),
        ('lasso', Lasso(alpha=50, max_iter=10000))
    ]
)
# pipe.fit(X_train, y_train)

param_grid = {
    'pca__n_components': [8]  # PCA 컴포넌트 수 조정
}

kfold = KFold(n_splits=15, random_state=321, shuffle=True)

grid_l = GridSearchCV(pipe, param_grid=param_grid, cv=kfold, scoring='neg_mean_squared_error')
grid_l.fit(X_train, y_train)

# 최적의 파라미터와 성능 출력
print("Best parameters:", grid_l.best_params_)
print("Best cross-validation score: {:.3f}".format(-grid_l.best_score_))


prediction = grid_l.predict(X_test)
get_evaluation(y_test, prediction)

MSE: 420.1151, RMSE: 20.4967, MSLE: 0.0010, RMSLE: 0.0314, R2: 0.7949
```
<img width="603" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/145a119d-1a9b-4844-b027-af16c1cd9a1a">
<img width="603" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/a4a1a4c3-7408-45c2-b0fb-071bd716139e">

<img width="665" alt="image" src="https://github.com/kimgusan/Machine_Learning/assets/156397911/2bd950a5-dc05-4320-b501-ac74b0f74dde">

<hr>

-   정리

    -   초기 분석에서 선형 회귀 모델을 사용하여 데이터를 훈련시켰을 때, 예상보다 높은 성능을 보였습니다.  
        이를 통해 데이터의 특성과 만족도 사이의 강한 선형 관계를 확인할 수 있었습니다.  
        그러나 모든 데이터가 유용한 것은 아니기 때문에 차원 축소를 통해 데이터의 희소성을 높이고 중요한 특성만을 강조했습니다.

    -   이어진 과정에서는 교차 검증을 활용해 모델의 과적합 정도를 점검하였습니다.  
        과적합을 관리하기 위해 L2 규제를 적용, R2 점수를 조절하여 모델의 일반화 능력을 강화했습니다.  
        이러한 훈련 모델은 새로운 데이터에 대해 더 잘 일반화하도록 설정되었습니다.

    - 또한 신뢰성과 일반화를 높이기 위해 최종적으로는 l2 규제를 사용하여 모델을 훈련시켰습니다.

-   결론

    -   회귀 분석을 통해 다양한 요소들과 만족도 점수 간의 관계를 식별할 수 있었습니다.  
        분석 결과, 긍정적 요소들은 만족도 점수와 높은 양의 상관관계를 가지는 반면, 삶의 만족도에 영향을 미치는 음수적 요소들은 상대적으로 영향력이 낮았습니다.  
        이는 긍정적인 요소들이 개인의 삶의 질을 높이는 데 중요한 역할을 한다는 것을 시사합니다.

    -   이러한 분석을 통해 얻은 모델은 개인이 웰빙 점수를 향상시키기 위한 전략을 수립하는 데 도움이 될 수 있습니다.
    -   또한, 이 결과는 건강 전문가들에게도 유용한 데이터를 제공하여, 보다 효과적인 건강, 웰빙 관련 공익 정책을 개발하는 기반을 마련할 수 있습니다.