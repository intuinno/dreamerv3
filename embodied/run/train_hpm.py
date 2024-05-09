import re
from collections import defaultdict
from functools import partial as bind

import embodied
import numpy as np


def train_hpm(make_agent, make_logger, args):

  agent = make_agent()
  logger = make_logger()

  logdir = embodied.Path(args.logdir)
  logdir.mkdir()
  print('Logdir', logdir)
  step = logger.step
  usage = embodied.Usage(**args.usage)
  agg = embodied.Agg()
  epstats = embodied.Agg()
  episodes = defaultdict(embodied.Agg)
  policy_fps = embodied.FPS()
  train_fps = embodied.FPS()

  batch_steps = args.batch_size * (args.batch_length - args.replay_context)
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.when.Clock(args.log_every)
  should_eval = embodied.when.Clock(args.eval_every)
  should_save = embodied.when.Clock(args.save_every)

  dataset_train = iter(agent.dataset(bind(
      replay.dataset, args.batch_size, args.batch_length)))
  dataset_report = iter(agent.dataset(bind(
      replay.dataset, args.batch_size, args.batch_length_eval)))
  carry = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def train_step(tran, worker):
    if len(replay) < args.batch_size or step < args.train_fill:
      return
    for _ in range(should_train(step)):
      with embodied.timer.section('dataset_next'):
        batch = next(dataset_train)
      outs, carry[0], mets = agent.train(batch, carry[0])
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      agg.add(mets, prefix='train')
  driver.on_step(train_step)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.replay = replay
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  should_save(step)  # Register that we just saved.

  print('Start training loop')
  policy = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  driver.reset(agent.init_policy)
  while step < args.steps:

    step.increment()
    
    driver(policy, steps=10)

    if should_eval(step) and len(replay):
      mets, _ = agent.report(next(dataset_report), carry_report)
      logger.add(mets, prefix='report')

    if should_log(step):
      logger.add(agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(embodied.timer.stats(), prefix='timer')
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.write()

    if should_save(step):
      checkpoint.save()

  logger.close()
