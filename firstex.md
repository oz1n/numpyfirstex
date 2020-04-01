```python
import numpy as np

a = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
print(a)
```

    [[ 1  2  3  4  5  6  7]
     [ 8  9 10 11 12 13 14]]



```python
a.shape
```




    (2, 7)




```python
a[1, 1:6:2]
```




    array([ 9, 11, 13])




```python
a[1,5] = 20
print(a)
```

    [[ 1  2  3  4  5  6  7]
     [ 8  9 10 11 12 20 14]]



```python
a[1, 1:7:2]
```




    array([ 9, 11, 20])




```python
a[:, 2] = [1,2]
a
```




    array([[ 1,  2,  1,  4,  5,  6,  7],
           [ 8,  9,  2, 11, 12, 20, 14]])




```python
np.zeros((2,3))
```




    array([[0., 0., 0.],
           [0., 0., 0.]])




```python
np.full((2,2),99,dtype="float32")
```




    array([[99., 99.],
           [99., 99.]], dtype=float32)




```python
np.full_like(a.shape,4)
```




    array([4, 4])




```python
np.random.rand(4,2)
```




    array([[0.45883029, 0.68633023],
           [0.46898642, 0.40586776],
           [0.46595753, 0.66262137],
           [0.86067246, 0.69605118]])




```python
np.random.random_sample(a.shape)
```




    array([[0.59595012, 0.7376513 , 0.92484457, 0.91008189, 0.49089563,
            0.16197942, 0.69575255],
           [0.97971155, 0.19584607, 0.88553152, 0.75259767, 0.79214128,
            0.41608724, 0.39484034]])




```python
np.random.randint(4,7,size=(3,3))
```




    array([[5, 5, 4],
           [6, 4, 6],
           [4, 6, 6]])




```python
arr = np.array([[1,2,3]])
r1 = np.repeat(arr,3,axis=1)
print(r1)
```

    [[1 1 1 2 2 2 3 3 3]]



```python
output = np.ones((5,5))
inner1 = np.zeros((3,3))
inner1[1,1] = 9 
output[1:4,1:4] = inner1
print(output)

```

    [[1. 1. 1. 1. 1.]
     [1. 0. 0. 0. 1.]
     [1. 0. 9. 0. 1.]
     [1. 0. 0. 0. 1.]
     [1. 1. 1. 1. 1.]]


##### Be Careful When Copying Array !!!


```python
a = np.array([1,2,3])
b = a.copy()
b[0] = 10
print(a)
print(b)
```

    [1 2 3]
    [10  2  3]


##### 線性代數的方法：1 乘法

##### ![image.png](attachment:image.png)

##### C11=A11B11+A12B21                           (2)

##### C12=A11B12+A12B22                           (3)

##### C21=A21B11+A22B21                           (4)

##### C22=A21B12+A22B22                           (5)


```python
a = np.array([[2,3],[2,2]])
b = np.array([[3,3],[2,4]])
c = np.matmul(a,b)
print(c)
```

    [[12 18]
     [10 14]]


##### 行列式的計算
###### ![image.png](attachment:image.png)


```python
d = np.array([[2,2],[3,4]])
print(d)
np.linalg.det(d)
```

    [[2 2]
     [3 4]]





    1.9999999999999998



## ----------------------------------------
### 數據統計
####  找到最小值和最大值


```python
stats = np.array([[1,2,3],[4,5,6]])
np.min(stats, axis=1)
#np.max(stats)
```




    array([1, 4])



##### 數組求和


```python
np.sum(stats)
```




    21




```python
np.sum(stats, axis=0)
```




    array([5, 7, 9])



##### 重新整理數組矩陣


```python
before = np.array([[1,2,3,4],[5,6,7,8]])
after = before.reshape((4,2))

```




    array([[1, 2],
           [3, 4],
           [5, 6],
           [7, 8]])



##### 垂直合併矩陣


```python
v1 = np.array([1,2,3,4])
v2 = np.array([5,6,7,8])
v3 = np.vstack([v1, v2, v2, v2])
v4 = np.vstack([v1,v2,v3,v3])

print(v4.shape)
print(v4)
```

    (10, 4)
    [[1 2 3 4]
     [5 6 7 8]
     [1 2 3 4]
     [5 6 7 8]
     [5 6 7 8]
     [5 6 7 8]
     [1 2 3 4]
     [5 6 7 8]
     [5 6 7 8]
     [5 6 7 8]]


##### 水平合併矩陣


```python
h1 = np.ones([2,4])
h2 = np.zeros([2,2])
h3 = np.hstack((h1, h2))
print(h3)
```

    [[1. 1. 1. 1. 0. 0.]
     [1. 1. 1. 1. 0. 0.]]


##### 導入文件中的數據


```python
datafile = np.genfromtxt('data.txt', delimiter = ',')
datafile = datafile.astype('int32')
datafile
```




    array([[  1,  13,  21,  11, 196,  75,   4,   3,  34,   6,   7,   8,   0,
              1,   2,   3,   4,   5],
           [  3,  42,  12,  33, 766,  75,   4,  55,   6,   4,   3,   4,   5,
              6,   7,   0,  11,  12],
           [  1,  22,  33,  11, 999,  11,   2,   1,  78,   0,   1,   2,   9,
              8,   7,   1,  76,  88]], dtype=int32)



##### 布爾值及進階索引


```python
datafile > 50
```




    array([[False, False, False, False,  True,  True, False, False, False,
            False, False, False, False, False, False, False, False, False],
           [False, False, False, False,  True,  True, False,  True, False,
            False, False, False, False, False, False, False, False, False],
           [False, False, False, False,  True, False, False, False,  True,
            False, False, False, False, False, False, False,  True,  True]])




```python
datafile[datafile > 50]
```




    array([196,  75, 766,  75,  55, 999,  78,  76,  88], dtype=int32)




```python
((datafile > 50) & (datafile < 100))
```




    array([[False, False, False, False, False,  True, False, False, False,
            False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False,  True, False,  True, False,
            False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False,  True,
            False, False, False, False, False, False, False,  True,  True]])



##### 用List索引矩陣


```python
a = np.array([1,2,3,4,5,6,7,8,9])
a[[1,2,8]]
```




    array([2, 3, 9])


