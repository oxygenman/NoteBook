## python property函数

 property有两个用法

总的来说property函数是用来构造可访问的属性的，用来封装geter和seter方法的，可以用它来访问私有变量，property函数的原型为property(fget=None,fset=None,fdel=None,doc= None)

- ### 两种用法

  - ```python
    class C:
    def __init__(self):
        #私有变量x
        self.__x=None
    def getx(self):
        return self.__x
    def setx(self, value):
        self.__x=value
    def delx(self):
        del self.__x
     x=property(getx,setx,delx,'')
    ```

    只需加上最后一句话，我们就可以方便的访问私有变量x

    ```python
    c=C()
    c.x=1000
    y=c.x
    del c.x
    ```

  - 第二种使用描述符来实现

  - ```python
    class C:
      def __init__(self):
        self.__x = None
        @property
        def x(self):
          return self.__x
        @x.setter
        def x(self, value):
          self.__x=value
        @x.deleter:
        def x(self):
          del self.__x
    ```

    这样就可以 访问和方法一一样访问属性了

    如果打算只设为只读，那就写一个函数就好了

    实际上，property函数不是一个真正的函数，它是其实例拥有很多特殊方法的类，涉及的方法是 \_\_get\_\_, \_\_set\_\_方法和\_\_delete\_\_. 实现了其中任何一个方法的对象就叫描述符。