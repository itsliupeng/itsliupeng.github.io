---
layout: post
title: "aop 切面编程 和 HDFS Spark scala helper "
categories: scala, aop, spark
---

本文介绍下自己写的 AOP helper，代码在 [https://github.com/itsliupeng/aop-helper](https://github.com/itsliupeng/aop-helper)

使用面向切面编程（AOP）可以在不侵入业务代码的情况下把一些监控代码逻辑添加进来。

之前在微博平台时，在特殊时期，一些上行接口需要封禁，不能让用户修改自己的一些个人信息，自己用 AOP 做了 Blocked 注解，通过开关（类似于 zookeeper 配置）来控制 Blocked 是否生效。代码架构非常简洁，生效和失效也非常及时


在小米代码耦合很严重，耗时记录都是与业务代码耦合在一起打点，用 AOP 写了 Profiling 注解。由于 Scala 也是要编程成 JVM bytecode，所以也支持对 Scala 的方法做注解，通过类名开始是否是 "$" 来判定是否是 Scala Object 类

```scala
    private static boolean isNotScalaObject(JoinPoint joinPoint) {
        return !joinPoint.getSignature().getDeclaringTypeName().endsWith("$");
    }
```

---- 

Scala 可以定义 implicit class 对原有类做隐式转换，从而让原有类拥有 implict class 中的方法，很方便地对类做扩展（Ruby 更灵活，可以直接打开类添加方法，也就是类的完整定义可以散布在多处，缺点是如果乱用，代码耦合会很痛苦，对阅读代码很不友好）

在 Spark.scala 中对 String 类型添加了 overwritePath， removePath， saveAsTextFileAndProduction 等方法，因为 Spark 默认的 save 方法不支持对文件进行覆盖。

在 Monitor.scala 中添加了对线程池和 guava cache 进行监控的方法。之前在微博做性能优化，发现 ExecutorService 中默认的 newCachedThreadPoolExecutor 没限制最大线程数, newFixedThreadPoolExecutor 没限制线程最大 Queue size，如果负载过高，都会占用过多内存，从而使服务崩溃，所以都不使用默认的实现，且都需要添加拒绝策略

在 HDFS.scala 中用闭包实现了 HDFS.read 方法，不需要自己再关闭文件，支持传入 f 方法对每行做格式化处理。

```scala
  def read[B](pathStr: String)(f: Stream[String] => B): Try[B] = {
    val path = new Path(pathStr)
    if (fileSystem.exists(path)) {
      val files = filesWithPrefix(path).map(p => new BufferedReader(new InputStreamReader(fileSystem.open(p.getPath))))
      val stream = files.map(br => Stream.continually(br.readLine()).takeWhile(_ != null)).foldLeft(Stream.empty[String])(_ ++ _)
      try {
        // f function must be eagerly evaluated before finally
        Success(f(stream))
      } finally {
        files.foreach(f => f.close())
      }
    } else {
      logger.error(s"path $pathStr is not existed")
      Failure(new InvalidPathException(s"path $pathStr is not existed"))
    }
  }
```

由于使用了 stream lazy 去读取，所以需要传入的处理方法 f 必须是 eagerly，不然如果是 lazy 的方式还没把文件内容读出来，finally 会把文件关闭