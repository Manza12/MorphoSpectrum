import logging as log


def configure_logs():
    log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='logs/main.log',
                    filemode='w')

    console = log.StreamHandler()
    console.setLevel(log.INFO)
    formatter = log.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    log.getLogger('').addHandler(console)
