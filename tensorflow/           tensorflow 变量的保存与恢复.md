#            tensorflow 变量的保存与恢复

tensorflow 变量的保存主要用到的类是：

```python
tf.train.Saver()
```

saver的构造必须放在变量初始化操作的后面，否则Saver()找不到需要保存的变量

Saver的三种构造方式：

```python
v1 = tf.Variable(..., name='v1')
v2 = tf.Variable(..., name='v2')

# Pass the variables as a dict:
saver = tf.train.Saver({'v1': v1, 'v2': v2})

# Or pass them as a list.
saver = tf.train.Saver([v1, v2])
# Passing a list is equivalent to passing a dict with the variable op names
# as keys:
saver = tf.train.Saver({v.op.name: v for v in [v1, v2]})
```

此处就涉及到tensorflow中变量名字的问题，TensorFlow构造变量的两种形式

```python
tf.Variable()
tf.get_variable()
```

两种方法的区别：目前我感觉差不多

但tensorflow中变量的名字都需要用name属性来制定，如不指定，tensorflow会自动个命名类似Variable,Variable_1等。

tf.Variable()方法会自动更改重名变量的名字 自动为它们加上下标。

猜测：在变量的restore过程中，程序根据Saver方法中定义的关系来恢复变量值。

实验证明是这样的，不出所料。