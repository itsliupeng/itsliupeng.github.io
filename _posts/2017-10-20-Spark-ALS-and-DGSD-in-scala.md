---
layout: post
title: "Matrix Factoration: Spark ALS and DSGA in Scala"
categories: JVM, performance
---

基于 Spark ALS 实现了增量式更新 userFactors， itemFacotors，支持加入新的 user 和 item，源码在[https://github.com/itsliupeng/incremental_spark_als_mf](https://github.com/itsliupeng/incremental_spark_als_mf)

基于论文 [ Large-scale matrix factorization with distributed stochastic gradient descent](https://dl.acm.org/citation.cfm?id=2020426) 使用 Scala 和 Spark 实现了 DSGD，源码在[https://github.com/itsliupeng/DSGD_spark](https://github.com/itsliupeng/DSGD_spark)

具体场景可参考下面 pdf

<iframe src="//www.slideshare.net/slideshow/embed_code/key/ICRZcoiN5K0Dcb" width="595" height="485" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" style="border:1px solid #CCC; border-width:1px; margin-bottom:5px; max-width: 100%;" allowfullscreen> </iframe> <div style="margin-bottom:5px"> <strong> <a href="//www.slideshare.net/pengliu5680/matrix-factoration-and-dsgd-in-spark" title="Matrix factoration and DSGD in Spark" target="_blank">Matrix factoration and DSGD in Spark</a> </strong> from <strong><a href="https://www.slideshare.net/pengliu5680" target="_blank">peng liu</a></strong> </div>
