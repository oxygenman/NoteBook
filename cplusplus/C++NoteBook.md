#                   c++笔记

[c++函数重载与重载原理：命名倾轧](https://blog.csdn.net/apollon_krj/article/details/60760586)

## 2

### 2.6引用

引用最直观的理解就是别名。

引用定义时必须赋值，引用不可以修改。

----------------------------

引用的精髓 主要用于传参

```c++
void swap(int &a, int &b)
{
    int t= a;
    a=b;
    b=t;
}//reference 本质就是对指针的包装
void swap(int *a, int *b)
//指针方式，裸露内存地址，并且需要开辟新的内存空间
```

引用的引用和引用的指针都不合法

引用的本质是常指针

const

```c++
double val =3.14
const int &ref = val;//开辟了一个未命名的空间（应该是放常量的全局空间），以只读的方式引用
//如果不用const的话，不会开辟新的空间所以不合法。
```

### 2.7 new/delete

### 2.8 内联函数（inline function）

宏函数/#define SQR(x) ((x)*(x))

优点: 内嵌到目标代码，减少了函数的调用

缺点：在预处理阶段完成替换，有可能带来语义上的差错

正常函数：

优点：完成了某一些操作的抽象，避免了相同功能的重复开发，有类型检查

缺点：无法避免压栈与出栈的开销



inline 关键字 结合了两者的优点

c++中的const是真constant,它是为了取代#define定义的宏常量

## 3命名空间

### 3.1命名空间

命名空间实际上是对全局作用域的划分

命名空间实际上就是一个由程序设计者命名的内存区域，程序设计者可以根据需要指定一些有名字的空间域，把一些全局实体分别放在各个命名空间中，从而与其他全局实体分割开来，避免命名冲突。

## 4封装 此处省略n字

今天安装了clion,顺便查了[make,cmake,gcc,clang的区别](https://oldpan.me/archives/gcc-make-cmake-clang-tell)

### 4.11 成员函数的存储方式

#### 4.11.1 类成员的组成

只用一段空间来存放共同的函数代码段，在调用各对象的函数时，都去调用这个公用的函数代码，c++编译系统怎么做呢？每个对象所占用的空间只是该对象的数据部分所占用的存储空间，而不包括函数代码所占用的存储空间。

### 4.12 const 

const 修饰成员变量，表示，不可改变。初始化只能形参列表里。

const 修饰成员函数，位置函数名以后，函数实现体以前。并且const可以构成重载

const承诺不会修改数据成员。const写在函数名后面。

const 修饰类对象obj,从对象层承诺不会修改数据成员。

只能调用const成员函数。

非const对象，优先调用非const成员函数，如果没有，有重载的const成员函数，也是选择。

### 4.15 static 修饰符

sttatic 不属于类对象，而属于类。

static 成员必须要初始化，初始化的方法，类型 类名：：变量 = 初值

static 数据存储在datarw

static 主要用于类对象之间共享数据用的

1. 静态成员函数的意义，不在于信息共享，数据沟通，而在于管理静态数据成员，完成对静态数据成员的封装。

2. 静态成员函数只能访问静态数据成员，原因：非静态成员函数，在调用this指针时被当做参数传进。而静态成员函数属于类，而不属于对象，没有this指针。

   ### 4.16 指向类成员的指针

   [为什么要使用构造函数初始化列表](https://blog.csdn.net/zizi7/article/details/52383015)

   [c++中new和java中new对象的区别](https://blog.csdn.net/wangfei8348/article/details/51383805)

   ## 5友元

   友元的本质目的，是让其它不属于本类的成员（全局函数，其它类的成员函数），成为类的成员而具备本类成员的属性。

   ## 6.继承 派生

   1继承方式 pubilc protected private

   共有继承：基类的公有成员和保护成员在派生类中保持原有的访问属性，其私有成员仍为基类的私有成员。

   私有继承：基类的公有成员和保护成员在派生类中成了私有成员，其私有成员仍为基类的私有成员

   保护继承：基类的公有成员和保护成员在派生类中成了保护成员，其私有成员仍为基类的私有成员

    [c++中拷贝构造函数的深拷贝和浅拷贝](https://blog.csdn.net/libin66/article/details/53140284)

   多继承

   多个负类中有同名的成员，被继承到一个类中，会给访问带来很大的不方便，浪费空间。

   解决多继承当中，同名成员被继承后引起访问混乱。

   z中只有一个data setData和getData 操作的是同一个data

   虚基类 virtual

   ## 8.多态

    纯虚函数

   1.没有实现体

   2.含有纯虚函数的类，叫抽象基类，abstract class 不能实例化

   3.如果子类没有复写，则子类也为抽象类

   凡是含有虚函数的类，其析构也要虚一下。可以实现完整的析构

   ​                                     

   