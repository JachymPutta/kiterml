#! /bin/lua
local res = ""

function getRes(filename)
  local handle = io.popen(string.format("../kiter/Release/bin/kiter -aKPeriodicThroughput -f %s", filename))
  res = handle:read("*a")
  handle:close()
end

function genGraph(cur, l,m, r)
  os.execute(string.format("./generate_lists.lua %d %d %d %d", cur, l,m, r))
end

function getFlow(l, m, r)
  local lo, hi, mid = 1,1000000,0

  while (lo <= hi) do
    mid = (lo+hi) // 2

    genGraph(mid, l, m ,r)
    getRes(string.format("lists/3/%d_%d_%d.xml", l, m, r))

    if string.find(res,"inf") then
      lo = mid+1
    else
      hi = mid-1
    end
  end

  while (string.find(res, "inf")) do
    mid = mid + 1

    genGraph(mid, l, m, r)
    getRes(string.format("lists/3/%d_%d_%d.xml", l, m, r))

    -- print("Additional loop is actually getting called?")
  end

  os.execute(string.format("echo %d >> results.txt", mid))
  -- print(string.format("Result for %d_%d_%d.xml : %d", l, m, r, mid))
end

for i = 1, 20, 1 do
  for j = 1, 20, 1 do
    for k = 1, 20, 1 do
      getFlow(i, j, k)
    -- genGraph(1000000, i, j)
    end
  end
end

-- getFlow(7, 2, 17)
-- genGraph(100000,3,2,17)



