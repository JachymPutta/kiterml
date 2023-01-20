#! /bin/lua

local LENGTH = 5
local RANGE = 5
local ACTOR_TYPE = 'A'
local INIT_TOKEN = arg[1] == nil and 10000 or arg[1]
-- local SEED = arg[2] == nil and 42 or arg[2]
local LEFT = arg[2]
local MID = arg[3]
local RIGHT = arg[4]

function printList(list, sep, file)
    file:write(table.concat(list, sep))
end

function genRandList(len, range)
    list = {}

    -- math.randomseed(math.floor(os.clock() * 1000000))
    -- math.randomseed(SEED)

    for i=1, len,1 do
        list[i] = math.random(1, range)
    end

    return list
end


function printActors(list, file)
    for i, el in ipairs(list) do
        file:write(string.format('\t\t\t<actor name="a%d" type="%s">\n', i, ACTOR_TYPE))
        file:write(string.format('\t\t\t\t<port name="in%d" type="in" rate="%d"/>\n', i, el))
        file:write(string.format('\t\t\t\t<port name="out%d" type="out" rate="%d"/>\n', i, el))
        file:write('\t\t\t</actor>\n')
    end
end

function printChannels(list, file)
    for i, el in ipairs(list) do
        if (i == #list)
        then
            file:write(string.format('\t\t\t<channel name="d%d" srcActor="a%d" srcPort="out%d" dstActor="a%d" dstPort="in%d" size="1" initialTokens="%d"/>\n',
            i, i, i, 1, 1, INIT_TOKEN))
        else
            file:write(string.format('\t\t\t<channel name="d%d" srcActor="a%d" srcPort="out%d" dstActor="a%d" dstPort="in%d" size="1" initialTokens="0"/>\n',
            i, i, i, i+1, i + 1))
        end
    end
end

function printProperties(list, file)
    for i, _ in ipairs(list) do
        file:write(string.format('\t\t\t<actorProperties actor="a%d">\n', i))
        file:write('\t\t\t\t<processor type="cluster_0" default="true">\n')
        file:write('\t\t\t\t\t<executionTime time="1"/>\n')
        file:write('\t\t\t\t</processor>\n')
        file:write(string.format('\t\t\t</actorProperties>\n'))
    end
end

function printGraph(list, file)
    -- file:write(HEADER)
    -- os.execute(string.format("echo %s >> lists.txt", table.concat(list,"")))
    file:write('<?xml version="1.0" encoding="UTF-8"?>\n')
    file:write('<sdf3 version="1.0" type="sdf" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n')
    file:write('\t<applicationGraph name="app">\n')

    file:write('\t\t<sdf name="tmp" type="Example">\n')
    printActors(list, file)
    printChannels(list, file)
    file:write('\t\t</sdf>\n')

    file:write('\t\t<sdfProperties>\n')
    printProperties(list, file)
    file:write('\t\t</sdfProperties>\n')

    file:write('\t</applicationGraph>\n')
    file:write('</sdf3>\n')
end

function genAllLists()
  local LIM_TOKEN = 20

  for left = 1, LIM_TOKEN, 1 do
    -- for mid = 1, LIM_TOKEN, 1 do
      for right = 1, LIM_TOKEN, 1 do
        file = io.open(string.format("./lists/2/%d_%d.xml", left, right), 'w')
        printGraph({left, right}, file)
        os.execute(string.format("echo %d,%d >> lists.txt", left, right))
        file:close()
      end
    -- end
  end

end
-- local list = genRandList(LENGTH, RANGE)


-- genAllLists()

function writeParticularList()
  file = io.open(string.format("./lists/3/%d_%d_%d.xml",LEFT, MID, RIGHT), 'w')
  printGraph({LEFT, MID, RIGHT}, file)
  os.execute(string.format("echo %d,%d,%d >> lists.txt",LEFT, MID, RIGHT))
  file:close()
end

writeParticularList()


