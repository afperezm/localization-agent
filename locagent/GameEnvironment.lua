-- Import relevant packages
py = require('fb.python')

-- The GameEnvironment class.
local gameEnv = torch.class('locagent.GameEnvironment')

--[[ Default class constructor.
]]
function gameEnv:__init(_opt)
  local _opt = _opt or {}
  self.verbose = _opt.verbose or 0
  self._state = {}
  self:_init(_opt.env, _opt.env_params, _opt.config_file)
  return self
end


--[[ Initializes the game environment.
]]
function gameEnv:_init(_env, _params, _config_file)

  local env = _env or 'localization_game'
  local config_file = _config_file or '/home/andresf/workspace-locagent/jointNetwork/debug/aeroplane0/rl.config'

  if self.verbose > 0 then
    print('\nPlaying:', env)
  end

  py.exec([[import learn.rl.RLConfig as config]])
  py.exec([[config.readConfiguration(configFile)]], {configFile = config_file})
  py.exec([[import ImageDraw]])
  py.exec([[import numpy]])
  py.exec([[from dotmap import DotMap]])
  py.exec([[from detection.boxsearch.BoxSearchEnvironment import BoxSearchEnvironment]])
  py.exec([[from detection.boxsearch.BoxSearchTask import BoxSearchTask]])
  py.exec([[import detection.boxsearch.BoxSearchState as BoxSearchState]])

  self._actions   = self:getActions()

  py.exec([[k = 0]])

  py.exec([[mode = 'train']])

  py.exec([[maxInteractions = config.geti(mode + 'Interactions')]])

  py.exec([[imageList = config.get(mode + 'Database')]])

  py.exec([[groundTruthFile = config.get(mode + 'GroundTruth')]])

  py.exec([[controller = DotMap()]])

py.exec([=[
def prepareImage(image):
  pass
]=])

py.exec([=[
def coverRegion(box):
  print('Covering region')
  w = box[2]-box[0]
  h = box[3]-box[1]
  b1 = map(int, [box[0] + w*0.5 - w*config.getf('markWidth'), box[1], box[0] + w*0.5 + w*config.getf('markWidth'), box[3]])
  b2 = map(int, [box[0], box[1] + h*0.5 - h*config.getf('markWidth'), box[2], box[1] + h*0.5 + h*config.getf('markWidth')])
  draw = ImageDraw.Draw(task.env.state.visibleImage)
  draw.rectangle(b1, fill=1)
  draw.rectangle(b2, fill=1)
  del draw
]=])

  py.exec([[controller.net.coverRegion = coverRegion]])
  py.exec([[controller.net.prepareImage = prepareImage]])

  py.exec([[environment = BoxSearchEnvironment(imageList, mode, controller, groundTruthFile)]])

  py.exec([[task = BoxSearchTask(environment, groundTruthFile)]])

  return self
end


--[[ Starts a new game and returns its state.
]]
function gameEnv:newGame()

  py.exec([[k = 0]])

  py.exec([[task.env.loadNextEpisode()]])

--  if training then
--    py.exec([[mode = 'train']])
--    py.exec([[maxInteractions = config.geti('trainInteractions')]])
--  else
--    py.exec([[mode = 'test']])
--    py.exec([[maxInteractions = config.geti('testInteractions')]])
--  end

  return self:getState()
end


--[[ Starts a new game by loading a new episode and returns its state.
]]
function gameEnv:nextRandomGame(k)
  return self:newGame()
end


--[[ Retrieves the current state of the game, represented by an observation
of the current game, the reward for the latest action taken, and a flag
telling whether the game has finished. 
]]
function gameEnv:getState()

  py.exec([[sensors = task.env.getSensors()]])
  py.exec([[cropped_image = task.env.state.visibleImage.crop(map(int,sensors['state'])).resize([50,50])]])

  self._state.observation = py.eval([[numpy.array(cropped_image.getdata()).reshape(cropped_image.size[0], cropped_image.size[1], 3)]])
  self._state.reward = py.eval([[task.getReward()]])
  self._state.terminal = py.eval([[task.env.episodeDone or k >= maxInteractions]])

  return self._state.observation, self._state.reward, self._state.terminal
end


--[[ Plays a given action in the game and returns the game state.
]]
function gameEnv:step(action, training)
  assert(action)

  py.exec([[task.performAction([actionChosen, float(actionValue)])]], {actionChosen = self._actions[action], actionValue = -1})

  py.exec([[k += 1]])

  return self:getState()
end


--[[ Returns the total number of pixels in one frame/observation
from the current game.
]]
function gameEnv:nObsFeature()
  return 50 * 50
end


--[[ Returns a table with valid game actions.
]]
function gameEnv:getActions()

  py.exec('actions = [BoxSearchState.X_COORD_UP]')
  py.exec('actions.append(BoxSearchState.Y_COORD_UP)')
  py.exec('actions.append(BoxSearchState.SCALE_UP)')
  py.exec('actions.append(BoxSearchState.ASPECT_RATIO_UP)')
  py.exec('actions.append(BoxSearchState.X_COORD_DOWN)')
  py.exec('actions.append(BoxSearchState.Y_COORD_DOWN)')
  py.exec('actions.append(BoxSearchState.SCALE_DOWN)')
  py.exec('actions.append(BoxSearchState.ASPECT_RATIO_DOWN)')
  py.exec('actions.append(BoxSearchState.PLACE_LANDMARK)')
  py.exec('actions.append(BoxSearchState.SKIP_REGION)')

  return py.eval('actions')
end
