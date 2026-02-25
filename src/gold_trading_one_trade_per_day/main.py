#!/usr/bin/env python
import sys
from gold_trading_one_trade_per_day.crew import GoldTradingOneTradePerDayCrew

# This main file is intended to be a way for your to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

import datetime
import os

def check_daily_trade_lock():
    lock_file = ".daily_trade_lock"
    today = datetime.date.today().isoformat()
    
    if os.path.exists(lock_file):
        with open(lock_file, "r") as f:
            last_trade_date = f.read().strip()
            if last_trade_date == today:
                return True
    return False

def set_daily_trade_lock():
    lock_file = ".daily_trade_lock"
    today = datetime.date.today().isoformat()
    with open(lock_file, "w") as f:
        f.write(today)

def run():
    """
    Run the crew.
    """
    if check_daily_trade_lock():
        print("DAILY TRADING LIMIT REACHED - SYSTEM HALTED (Lock file detected)")
        return

    inputs = {
        'daily_trade_executed': 'False'
    }
    result = GoldTradingOneTradePerDayCrew().crew().kickoff(inputs=inputs)
    
    # Check if a trade was actually executed before applying the lock
    result_str = str(result)
    if '"status": "Executed"' in result_str or "'status': 'Executed'" in result_str:
        set_daily_trade_lock()
        print("Trade executed successfully. Daily lock set.")
    else:
        print("No trade executed. Lock not set. Ready to shadow trade again.")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'daily_trade_executed': 'sample_value'
    }
    try:
        GoldTradingOneTradePerDayCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        GoldTradingOneTradePerDayCrew().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        'daily_trade_executed': 'sample_value'
    }
    try:
        GoldTradingOneTradePerDayCrew().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: main.py <command> [<args>]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "run":
        run()
    elif command == "train":
        train()
    elif command == "replay":
        replay()
    elif command == "test":
        test()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
