# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 13:34:00 2021

@author: saty
"""

import nest_asyncio
import uvicorn
import os
from main import app 
# Allows the server to be run in this interactive environment
nest_asyncio.apply()

# Host depends on the setup you selected (docker or virtual env)
host = "0.0.0.0" if os.getenv("DOCKER-SETUP") else "127.0.0.1"

# Spin up the server!    
uvicorn.run(app, host=host, port=8000)