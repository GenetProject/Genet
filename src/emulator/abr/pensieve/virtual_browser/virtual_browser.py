import argparse
from json import dumps
import multiprocessing as mp
import os
import signal
from time import sleep
from urllib.parse import ParseResult, parse_qsl, unquote, urlencode, urlparse

from pyvirtualdisplay import Display
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

from pensieve.virtual_browser.abr_server import run_abr_server

# TO RUN: download https://pypi.python.org/packages/source/s/selenium/selenium-2.39.0.tar.gz
# run sudo apt-get install python-setuptools
# run sudo apt-get install xvfb
# after untar, run sudo python setup.py install
# follow directions here: https://pypi.python.org/pypi/PyVirtualDisplay to install pyvirtualdisplay

# For chrome, need chrome driver: https://code.google.com/p/selenium/wiki/ChromeDriver
# chromedriver variable should be path to the chromedriver
# the default location for firefox is /usr/bin/firefox and chrome binary is /usr/bin/google-chrome
# if they are at those locations, don't need to specify

# ABR_ID_MAP = {
#     'Default': 0,
#     'FixedRate': 1,
#     'BufferBased': 2,
#     'RateBased': 3,
#     'RL': 4,
#     'RobustMPC': 4,
#     'Festive': 5,
#     'Bola': 6
# }


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Virtual Browser")
    parser.add_argument('--description', type=str, default=None,
                        help='Optional description of the experiment.')

    # ABR related
    parser.add_argument('--abr', type=str, required=True,
                        choices=['RobustMPC', 'RL', 'FastMPCchromedriver'
                                 'Default', 'FixedRate',
                                 'BufferBased', 'RateBased', 'Festive',
                                 'Bola'], help='ABR algorithm.')
    parser.add_argument('--actor-path', type=str, default=None,
                        help='Path to RL model.')

    # data io related
    parser.add_argument('--summary-dir', type=str, required=True,
                        help='directory to save logs.')
    parser.add_argument('--trace-file', type=str, required=True,
                        help='Path to trace file.')
    parser.add_argument("--video-size-file-dir", type=str, required=True,
                        help='Dir to video size files')
    parser.add_argument('--run_time', type=int, default=240,
                        help="Running time.")

    # networking related
    parser.add_argument('--ip', type=str, help='IP of HTTP video server.')
    parser.add_argument('--port', type=int,
                        help='Port number of HTTP video server.')
    parser.add_argument('--abr-server-ip', type=str, default='localhost',
                        help='IP of ABR server.')
    parser.add_argument('--abr-server-port', type=int, default=8333,
                        help='Port number of ABR server.')

    parser.add_argument('--buffer-threshold', type=int, default=60,
            help='Buffer threshold of Dash.js MediaPlayer. Unit: Second.')

    return parser.parse_args()


def add_url_params(url, params):
    """Add GET params to provided URL being aware of existing.

    url = 'http://stackoverflow.com/test?answers=true'
    new_params = {'answers': False, 'data': ['some','values']}
    add_url_params(url, new_params)
    'http://stackoverflow.com/test?data=some&data=values&answers=false'

    Args
        url: string of target URL
        params: dict containing requested params to be added

    Return
        string with updated URL
    """
    # Unquoting URL first so we don't loose existing args
    url = unquote(url)
    # Extracting url info
    parsed_url = urlparse(url)
    # Extracting URL arguments from parsed URL
    get_args = parsed_url.query
    # Converting URL arguments to dict
    parsed_get_args = dict(parse_qsl(get_args))
    # Merging URL arguments dict with new params
    parsed_get_args.update(params)

    # Bool and Dict values should be converted to json-friendly values
    # you may throw this part away if you don't like it :)
    parsed_get_args.update(
        {k: dumps(v) for k, v in parsed_get_args.items()
         if isinstance(v, (bool, dict))}
    )

    # Converting URL argument to proper query string
    encoded_get_args = urlencode(parsed_get_args, doseq=True)
    # Creating new parsed result object based on provided with new
    # URL arguments. Same thing happens inside of urlparse.
    new_url = ParseResult(
        parsed_url.scheme, parsed_url.netloc, parsed_url.path,
        parsed_url.params, encoded_get_args, parsed_url.fragment
    ).geturl()

    return new_url


def timeout_handler(signum, frame):
    raise Exception("Timeout")


def main():
    args = parse_args()
    ip = args.ip
    port_number = args.port
    abr_algo = args.abr
    run_time = args.run_time

    # start abr server here
    # prevent multiple process from being synchronized
    abr_server_proc = mp.Process(target=run_abr_server, args=(
        abr_algo, args.trace_file, args.summary_dir, args.actor_path,
        args.video_size_file_dir, args.abr_server_ip, args.abr_server_port))
    abr_server_proc.start()

    sleep(0.5)

    # generate url
    url = 'http://{}:{}/index.html'.format(ip, port_number)
    # url_params = {'abr_id': ABR_ID_MAP[abr_algo]}
    url_params = {'abr_id': abr_algo,
                  'buffer_threshold': args.buffer_threshold,
                  'port': args.abr_server_port}
    url = add_url_params(url, url_params)

    # ip = json.loads(urlopen("http://ip.jsontest.com/").read().decode('utf-8'))['ip']
    # url = 'http://{}/myindex_{}.html'.format(ip, abr_algo)
    print('Open', url)

    # timeout signal
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(run_time + 30)
    display = None
    driver = None
    try:
        # copy over the chrome user dir
        default_chrome_user_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'abr_browser_dir/chrome_data_dir')
        # chrome_user_dir = '/tmp/chrome_user_dir_id_' + process_id
        chrome_user_dir = '/tmp/lesley_chrome_user_dir'  # + process_id
        os.system('rm -r ' + chrome_user_dir)
        os.system('cp -r ' + default_chrome_user_dir + ' ' + chrome_user_dir)

        # to not display the page in browser
        display = Display(visible=False, size=(800, 600))
        display.start()

        # initialize chrome driver
        options = Options()
        chrome_driver = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'abr_browser_dir/chromedriver')
        options.add_argument('--user-data-dir=' + chrome_user_dir)
        # enable browser logging
        options.add_argument("--headless")
        options.add_argument("--disable-extensions")
        options.add_argument('--ignore-certificate-errors')
        options.add_argument( "--disable-web-security" )
        options.add_argument( "--disable-site-isolation-trials" )
        desired_caps = DesiredCapabilities.CHROME
        desired_caps['goog:loggingPrefs'] = {'browser': 'ALL'}
        #import pdb; pdb.set_trace()
        driver = webdriver.Chrome(chrome_driver, options=options,
                                  desired_capabilities=desired_caps)

        # run chrome
        driver.set_page_load_timeout(10)
        driver.get(url)

        sleep(run_time)
        driver.quit()
        display.stop()

        print('done')

    except Exception as e:
        if display is not None:
            display.stop()
        if driver is not None:
            driver.quit()
        # try:
        #     proc.send_signal(signal.SIGINT)
        # except:
        #     pass
        print(e)
    abr_server_proc.terminate()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard Interrupted! Virtual browser exits!')
