An example of using thimbleberry [benchmarking utilities](https://github.com/mighdoll/thimbleberry/blob/main/doc/Utilities.md#GPU-Performance-Reports) 
for a simple reduce shader by @muyyatin.

Run with:

```
pnpm i
pnpm vite 
```

Launch chrome on a mac with:
```
open -a "Google Chrome Canary" --args --enable-dawn-features=allow_unsafe_apis --disable-dawn-features=timestamp_quantization
```

View results in the browser dev console.

With default settings `benchRunner` runs the benchmark 100 times
and reports a summary of all benchmark runs 
and the details of the median fastest run.

`benchRunner` is configurable via url parameters, e.g.
```
http://localhost:5173?precision=4&runs=200&reportType=fastest
```