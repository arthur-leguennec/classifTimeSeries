cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-filename', '123', 'initial random seed')
params = cmd:parse(arg)

print(arg)
