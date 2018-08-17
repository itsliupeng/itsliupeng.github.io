---
layout: post
title: "Enum.hashCode is only consistent within the same JVM "
categories: java, JVM
---

### 问题起源 

画报一个 RDD[((String, com.xiaomi.data.spec.log.profile.o2o.ReachType), Long)] 进行 reduceByKey ，发现结果中 key 不是唯一的，不明白为什么 reduceByKey 没有把 hashCode 相同(其实 hashCode 不同)的 key 聚在一起

### 原因

[https://issues.apache.org/jira/browse/SPARK-3847](https://issues.apache.org/jira/browse/SPARK-3847)

Enum.hashCode is only consistent within the same JVM.

Java arrays' hashCodes have a similar problem: they are based on the arrays' identities rather than their contents.

### 结论

Enum 和 Array 的 hashCode 不与其内容一致（同一个 JVM 进程相同内容的 Enum 是一致的，相同内容的 Array 肯定不一致）
不用使用 Enum 和 Array 类型作为 PairRDDFunctions 中 reduceByKey, groupByKey, aggregateByKey 等聚合操作的 key


