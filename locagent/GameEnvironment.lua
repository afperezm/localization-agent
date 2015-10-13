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


--[[
]]
function gameEnv:_init(_env, _params, _config_file)

  local env = _env or 'localization_game'
  local config_file = _config_file or '/home/andresf/workspace-locagent/jointNetwork/debug/aeroplane0/rl.config'

  if self.verbose > 0 then
    print('\nPlaying:', env)
  end

  py.exec([[import learn.rl.RLConfig as config]])
  py.exec([[config.readConfiguration(configFile)]], {configFile = config_file})

  self._actions   = self:getActions()

  py.exec([[k = 0]])

  py.exec([[mode = 'train']])

  py.exec([[maxInteractions = config.geti(mode + 'Interactions')]])

  py.exec([[imageList = config.get(mode + 'Database')]])

  py.exec([[groundTruthFile = config.get(mode + 'GroundTruth')]])

  py.exec([[from dotmap import DotMap]])
  py.exec([[controller = DotMap()]])
  py.exec([[controller.net = None]])

  py.exec([[from detection.boxsearch.BoxSearchEnvironment import BoxSearchEnvironment]])
  py.exec([[environment = BoxSearchEnvironment(imageList, mode, controller, groundTruthFile)]])

  py.exec([[from detection.boxsearch.BoxSearchTask import BoxSearchTask]])
  py.exec([[task = BoxSearchTask(environment, groundTruthFile)]])

  return self
end


--[[ Function advances the emulator state until a new game starts and returns
this state. The new game may be a different one, in the sense that playing back
the exact same sequence of actions will result in different outcomes.
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


--[[ Function advances the emulator state until a new (random) game starts and
returns this state.
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
  py.exec([[image = task.env.state.visibleImage]])
  py.exec([[cropped_image = image.crop(map(int,sensors['state'])).resize([50,50])]])

  py.exec([[import numpy]])

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

  py.exec([[import detection.boxsearch.BoxSearchState as BoxSearchState]])

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
