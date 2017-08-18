---
layout: post
title: "Redis cluter JedisClusterMaxRedirectionsException (Too many Cluster redirections)"
categories: cache
---


好久没有写博客了，已经到小米工作有半年了

最近在做基于 ALS 矩阵分解的协同过滤。模型训练完成，生成推荐结果，在将其写入到 redis 时，spark job 报错 `redis.clients.jedis.exceptions.JedisClusterMaxRedirectionsException (Too many Cluster redirections)`

看了下源码，将 maxRetries 由  3 改为 6，还是同样错误。
最终还是将 spark executor num 由 30 降为 10，虽然本来 10 min 写完的任务延长到 30 min，但是能成功了

具体详细解释参考 http://carlosfu.iteye.com/blog/2251034。猜想原因应该是作为 DB 使用的 redis cluster AOF 太慢造成的。

现在工作重心由之前的片架构方面转到偏模型方面，再加上整个平台的氛围，没有心情去联系同事找到本质原因了，姑且记之 

