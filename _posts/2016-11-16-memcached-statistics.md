---
layout: post
title: "memcached statistics"
categories: cache
---

- `stats cachedump <slab_num> <number>`
- `stats slabs`
- `stats items`


#### slab page chunk
[http://www.mikeperham.com/2009/06/22/slabs-pages-chunks-and-memcached/](http://www.mikeperham.com/2009/06/22/slabs-pages-chunks-and-memcached/)

#### 钙化问题
在 1.4.11 版本加入 slab reassign, 从chunk 较小的 slab 中回收空闲内存分配给 chunk 较大的 slab
[https://github.com/memcached/memcached/wiki/ReleaseNotes1411](https://github.com/memcached/memcached/wiki/ReleaseNotes1411)
