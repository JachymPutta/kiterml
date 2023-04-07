#! /bin/lua
require 'const'

local res = ""


local function getRes(filename)
  local handle = io.popen(string.format("%s -aKPeriodicThroughput -f %s", KITER_PATH, filename))
  res = handle:read("*a")
  handle:close()
end

local function genGraph(cur, one, two, three, four)
  os.execute(string.format("./generate_lists.lua %d %d %d %d %d", cur, one, two, three, four))
end

local function getFlow(one, two, three, four)
  local lo, hi, mid = 1, 500, 0

  while (lo <= hi) do
    mid = (lo+hi) // 2

    genGraph(mid, one, two ,three, four)
    getRes(string.format(GRAPH_TYPE, one, two, three, four))

    if string.find(res,"inf") then
      lo = mid+1
    else
      hi = mid-1
    end
  end

  while (string.find(res, "inf")) do
    mid = mid + 1

    genGraph(mid, one, two, three, four)
    getRes(string.format(GRAPH_TYPE, one, two, three, four))

    -- print("Additional loop is actually getting called?")
  end

  os.execute(string.format("echo %d %d %d %d %d >> results.txt", one, two, three, four, mid))
  -- print(string.format("Result for %d_%d_%d.xml : %d", l, m, r, mid))
end

local function getFlowForRange(lo, hi)
  for i = lo, hi, 1 do
    for j = 1, LIM_TOKEN, 1 do
      for k = 1, LIM_TOKEN, 1 do
        for l = 1, LIM_TOKEN, 1 do
          getFlow(i, j, k, l)
          -- genGraph(1000000, i, j)
        end
      end
    end
  end
end

getFlowForRange(arg[1], arg[2])



