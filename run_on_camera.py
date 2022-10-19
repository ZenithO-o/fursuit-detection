import os
import sys
import getopt
import time
import cv2
import numpy as np

# Removes tensorflow logs (they're annoying to me 3:<)
# Comment out the line below if you actually want to see that stuff
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(argv: list[str]):
    """Script for running fursuit model on system camera (if exists)

    Args:
        argv (list[str]): system args
    """
    
    VERBOSE = False
    FULLSCREEN = False
    T = 0.5
    
    help_text = '''Run the fursuit model with your default system's camera!
Note: Press 'Q' to quit viewing\n\n'''
    arg_help = '''Usage: 
    python run_with_camera.py [options] [device]
  
Arguments:
    -h  --help
        how to use this script
    -v --verbose
        logs extra details for your command line to suffer (and maybe a few wahs)
    -f --fullscreen
        enable fullscreen view of camera output
    -t --threshold [float]
        the confidence percentage at which a bounding box will be 
        defined as a positive result (default=0.5, bounds=[0.,1.])
    '''
    
    try:
        opts, args = getopt.getopt(argv[1:],"hvft:",["help", "verbose", "fullscreen", "threshold"])
    except getopt.GetoptError:
        arg_help = "Error: I cannot understand you :(\n\n" + arg_help
        print(arg_help)
        sys.exit(2)
    
    # Option specific logic
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            arg_help = help_text + arg_help
            print(arg_help)
            sys.exit(2)
        
        elif opt in ('-v', '--verbose'):
            VERBOSE = True
        
        elif opt in ('-f', '--fullscreen'):
            FULLSCREEN = True
        
        elif opt in ('-t', '--threshold'):
            try:
                T = float( arg )
            except ValueError:
                arg_help = "Threshold must be a float\n\n" + arg_help
                print(arg_help)
                sys.exit(2)
            
            if T < 0 or T > 1:
                arg_help = "Threshold must be between [0.,1.]\n\n" + arg_help
                print(arg_help)
                sys.exit(2)
    
        # Checks if any additional arguments are passed
    if len(args) > 1:
        arg_help = "Error: Too many devices passed into args (Only 1 allowed)\n\n" + arg_help
        print(arg_help)
        sys.exit(2)
    elif len(args) != 0:
        try:
            device = int(args[0])
        except ValueError:
            arg_help = "Error: Device value incorrect (must be int)\n\n" + arg_help
            print(arg_help)
            sys.exit(2)
    else:
        device = 0
    
    if VERBOSE:
        print(f"Fullscreen: {FULLSCREEN}")
        print(f"Threshold: {T}")
        print('-'*15)

    if VERBOSE: print("Loading model...")

    start = time.perf_counter()

    from fursuit_model.model import FursuitModel
    model = FursuitModel()
    
    elapsed = time.perf_counter() - start
    
    if VERBOSE: print(f"Finished loading model! (time elapsed: {round(elapsed,2)})")
    if VERBOSE: print('-'*15)
    
    # Get capture
    cap = cv2.VideoCapture(device)
    cap.set(3, 1920)
    cap.set(3, 1080)
    # Sets window to fullscreen :)
    if FULLSCREEN:
        cv2.namedWindow('Fursuit Detector', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Fursuit Detector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        np_frame = np.array(frame)
        try:
            result = model.run_model(np_frame)
            model.visualize_detections(np_frame, result, T, True)
        except ValueError:
            print('An image was not detected from your camera. Please make sure your settings are correct!')
            break
        cv2.imshow('Fursuit Detector', np_frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
        else:
            pass


if __name__ == "__main__":
    main(sys.argv)