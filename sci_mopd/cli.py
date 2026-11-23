"""Unified entry point; each command is also available as python -m sci_mopd.<command>."""
import argparse,importlib,sys

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['data','sft','train','infer','evaluate','summarize','smoke','shortcuts']);p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
 if a.command=='data':
  sys.argv=['sci_mopd.data']+a.arguments;import runpy;runpy.run_module('sci_mopd.data',run_name='__main__')
 else:importlib.import_module('sci_mopd.'+a.command).main(a.arguments)
if __name__=='__main__':main()
