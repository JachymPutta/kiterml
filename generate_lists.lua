#! /bin/lua

require 'const'

local function printActors(list, file)
    for i, el in ipairs(list) do
        file:write(string.format('\t\t\t<actor name="a%d" type="%s">\n', i, ACTOR_TYPE))
        file:write(string.format('\t\t\t\t<port name="in%d" type="in" rate="%d"/>\n', i, el))
        file:write(string.format('\t\t\t\t<port name="out%d" type="out" rate="%d"/>\n', i, el))
        file:write('\t\t\t</actor>\n')
    end
end

local function printChannels(list, file)
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

local function printProperties(list, file)
    for i, _ in ipairs(list) do
        file:write(string.format('\t\t\t<actorProperties actor="a%d">\n', i))
        file:write('\t\t\t\t<processor type="cluster_0" default="true">\n')
        file:write('\t\t\t\t\t<executionTime time="1"/>\n')
        file:write('\t\t\t\t</processor>\n')
        file:write(string.format('\t\t\t</actorProperties>\n'))
    end
end

local function printGraph(list, file)
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

local function genAllLists()

  for one = 1, LIM_TOKEN, 1 do
  for two = 1, LIM_TOKEN, 1 do
        for three = 1, LIM_TOKEN, 1 do
            for four = 1, LIM_TOKEN, 1 do
                fileName = string.format(GRAPH_TYPE, one, two, three, four)
                print(fileName)
                file = io.open(fileName, 'w')
                printGraph({one, two, three, four}, file)
                os.execute(string.format("echo %d,%d,%d,%d >> lists.txt", one, two, three, four))
                file:close()
            end
        end
    end
  end

end

local function writeParticularList()
  file = io.open(string.format(GRAPH_TYPE, ONE, TWO, THREE, FOUR), 'w')
  printGraph({ONE, TWO, THREE, FOUR}, file)
  os.execute(string.format("echo %d,%d,%d,%d >> lists.txt",ONE, TWO, THREE, FOUR))
  file:close()
end

if (ONE) then 
    writeParticularList()
else 
    genAllLists()
end


