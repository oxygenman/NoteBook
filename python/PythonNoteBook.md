幂操作运算符**有点儿特殊。

Python可以这样3<4<5

断言 assert

For循环

For each in 集合

range()

range([start,][stop][step])

range和for循环经常搭配使用

列表很强大可以添加各种元素

append()只能添加一个参数，也可以添加列表，但显示的就是一个列表

extend()的参数是一个列表

列表类型的内置函数

List的复制

List1=list[:]这是copy

元组：是戴上了枷锁的列表 tuple

Temp的元素添加和改进

Format

List(iterrable) 将一个可迭代对象转换为列表

2 函数，python乐高积木

注意函数的默认参数

收集参数def test(*params):

内部函数 nonlocal global 关键字

Lamda 匿名函数

过滤器Filter函数将

3递归 调用函数自身 有停止条件

4字典用大括号表示{} 是映射类型

Dict.fromkeys()

5.集合是无序的 set里面的元素都是唯一的

6.文件的写入

7.OS可以让你不用管是哪个操作系统 哪个编程语言

8.腌菜 pickle 模块

9面向对象编程 封装 继承 多态

 Self代表自身

Python 中有类对象和实例对象 实例对象中的方法会覆盖类对象中的方法

相同名字的属性会覆盖方法

 Python中要绑定（self）

10 .魔法方法 魔法方法一般__xxxx__

 不要试图对__init__写返回值

11.垃圾回收机制会调用__del__进行析构

12.访问属性的魔法方法 使用魔法方法的时候可能会陷入死循环的陷阱

--------------------------------------------------------------------------------------------------------------------------------------

#                                python学习笔记

1. ## python的命令行输入

   python读取命令行的输入最简单的方法是使用input()和raw_input()函数

   两者的区别，例子：

   ```python
   name=intput("what is your name?")
   print ("hello,"+name+"!")
   ```

    这个程序会有什么问题呢？

   当你的输入是 xy 而不是"xy"时，程序就会报错，因为input默认你的输入是合法的python表达式,然后输出的时候会把引号去掉输出。

   而raw_input()会把所有的数据当做raw data,就是python不会对它做任何处理

   python3中只有input替代了raw_input()

   ```python
   input("Enter a number:")
   Enter a number:3
   3
   raw_input("Enter a number:")
   Enter a number:3
   "3"
   ```

   ​