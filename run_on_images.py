from email.mime import image
import os
import sys
import getopt
import time
import filetype

# Removes tensorflow logs (they're annoying to me 3:<)
# Comment out the line below if you actually want to see that stuff
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(argv: list[str]):
    """Script for running fursuit detection model on images.

    Args:
        argv (list[str]): system args
    """
    
    VERBOSE = False
    DISPLAY = False
    CROPPED = False
    EXPORT_DIR = None
    T = 0.5

    arg_help = '''Usage:
    python run_on_images.py [options] [file ...]
  
Arguments:
    -h  --help
        heheheha
    -v  --verbose
        logs extra details for you command line to suffer (and maybe a few wahs)
    -d  --display
        opens the detected image results in default OS viewer
    -c  --crop
        shows only cropped boxes, as opposed to boxes' labels overlayed on image
    -e  --export-results [path]
        saves results to path specified
    -t  --threshold [float]
        the confidence percentage at which a bounding box will be 
        defined as a positive result (default=0.5, bounds=[0.,1.])
    '''


    # Parses into options and args
    try:
        opts, args = getopt.getopt(argv[1:],"hvdce:t:",["help", "verbose", "display", "crop", "export-results", "threshold"])
    except:
        arg_help = "Error: I cannot understand you :(\n\n" + arg_help
        print(arg_help)
        sys.exit(2)

    # Checks if any additional arguments are passed
    if len(args) == 0:
        arg_help = "Error: Please input an image\n\n" + arg_help
        print(arg_help)
        sys.exit(2)

    # Option specific logic
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(arg_help)
            sys.exit(2)

        elif opt in ('-v', '--verbose'):
            VERBOSE = True

        elif opt in ('-d', '--display'):
            DISPLAY = True

        elif opt in ('-c', '--crop'):
            CROPPED = True

        elif opt in ('-e', '--export-results'):
            EXPORT_DIR = arg
            if not os.path.isdir(arg):
                arg_help = "Must provide a valid export path\n\n" + arg_help
                print(arg_help)
                sys.exit(2)

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


    # Checks if args are image files 
    # (does not guarantee that the images are valid formats themselves)
    for arg in args:
        if not filetype.is_image(arg):
            arg_help = f"{arg} is not a valid file\n\n" + arg_help
            print(arg_help)
            sys.exit(2)

    if VERBOSE:
        print(f"Display: {DISPLAY}")
        print(f"Cropped: {CROPPED}")
        print(f"Export Dir: {EXPORT_DIR}")
        print(f"Threshold: {T}")
        print('-'*15)


    # Opening the images
    import numpy as np
    from PIL import Image
    images = []
    for arg in args:
        img = Image.open(arg)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.array(img)
        images.append(img)
    
    if VERBOSE: print("Loading model...")

    start = time.time()
    
    from fursuit_model.model import FursuitModel
    model = FursuitModel()
    
    elapsed = time.time() - start
    
    if VERBOSE: print(f"Finished loading model! (time elapsed: {round(elapsed,2)})")
    if VERBOSE: print('-'*15)
    
    for i in range(len(images)):
        if VERBOSE: print(f'Running model on `{args[i]}`...')

        # Running model on images
        start = time.time()

        result = model.run_model(images[i])
        num_detections = len([score for score in result['detection_scores'] if score >= T])

        elapsed = time.time() - start

        if VERBOSE: print(f'Wah! Found {num_detections} bounding boxes in {args[i]} (time elapsed: {round(elapsed,2)})')
        
        #Convert to visualization or cropped
        image_results = []
        if not CROPPED:
            image_result = model.visualize_detections(images[i], result, T)
            image_results = [image_result]
        else:
            image_results, labels = model.crop_detections(images[i], result, T)

        # Dispalying model with imshow
        if DISPLAY:
            if VERBOSE: print("Displaying image...")
            
            count = 0
            for image in image_results:
                img = Image.fromarray(image)
                img.show(f"{args[i]}_{count}")
                count += 1


        # Save image
        if EXPORT_DIR:
            count = 0
            for image in image_results:
                img = Image.fromarray(image)
                
                # oh god
                file_name = args[i][args[i].rfind('\\')+1:args[i].rfind('.')]
                save_path = os.path.join(EXPORT_DIR, f'{file_name}_{count}.png')
                
                if VERBOSE: print(f"Saving image to `{save_path}`...")
                img.save(save_path)
                
                count += 1

        if VERBOSE: print('-'*15)

if __name__ == "__main__":
    main(sys.argv)