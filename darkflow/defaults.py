class argHandler(dict):
    #A super duper fancy custom made CLI argument handler!!
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    _descriptions = {"help, -h": "show this super helpful message and exit"}
    
    def define(self, argName, default, description):
        self[argName] = default
        self._descriptions[argName] = description
    
    def help(self):
        print("Example usage: flow --imgdir sample_img/ --model cfg/yolo.cfg --load bin/yolo.weights")
        print()
        print("Arguments:")
        spacing = max([len(i) for i in self._descriptions.keys()]) + 2
        for item in self._descriptions:
            currentSpacing = spacing - len(item)
            print("  --" + item + (" " * currentSpacing) + self._descriptions[item])
        print()
        exit()

    def parseArgs(self, args):
        print()
        i = 1
        while i < len(args):
            if args[i] == "-h" or args[i] == "--help":
                self.help() #Time for some self help! :)
            if len(args[i]) < 2:
                print("ERROR - Invalid argument: " + args[i])
                exit()
            argumentName = args[i][2:]
            if isinstance(self.get(argumentName), bool):
                if not (i + 1) >= len(args) and (args[i + 1].lower() != "false" and args[i + 1].lower() != "true") and not args[i + 1].startswith("--"):
                    print("ERROR - Expected boolean value (or no value) following argument: " + args[i])
                elif not (i + 1) >= len(args) and (args[i + 1].lower() == "false" or args[i + 1].lower() == "true"):
                    self[argumentName] = (args[i + 1].lower() == "true")
                    i += 1
                else:
                    self[argumentName] = True
            elif args[i].startswith("--") and not (i + 1) >= len(args) and not args[i + 1].startswith("--") and argumentName in self:
                self[argumentName] = args[i + 1]
                i += 1
            else:
                print("ERROR - Invalid argument: " + args[i])
                exit()
            i += 1

defaultFLAGS = argHandler()

defaultFLAGS.define("imgdir", "./sample_img/", "path to testing directory with images")
defaultFLAGS.define("binary", "./bin/", "path to .weights directory")
defaultFLAGS.define("config", "./cfg/", "path to .cfg directory")
defaultFLAGS.define("dataset", "../pascal/VOCdevkit/IMG/", "path to dataset directory")
defaultFLAGS.define("backup", "./ckpt/", "path to backup folder")
defaultFLAGS.define("summary", "./summary/", "path to TensorBoard summaries directory")
defaultFLAGS.define("annotation", "../pascal/VOCdevkit/ANN/", "path to annotation directory")
defaultFLAGS.define("threshold", -0.1, "detection threshold")
defaultFLAGS.define("model", "", "configuration of choice")
defaultFLAGS.define("trainer", "rmsprop", "training algorithm")
defaultFLAGS.define("momentum", 0.0, "applicable for rmsprop and momentum optimizers")
defaultFLAGS.define("verbalise", True, "say out loud while building graph")
defaultFLAGS.define("train", False, "train the whole net")
defaultFLAGS.define("load", "", "how to initialize the net? Either from .weights or a checkpoint, or even from scratch")
defaultFLAGS.define("savepb", False, "save net and weight to a .pb file")
defaultFLAGS.define("gpu", 0.0, "how much gpu (from 0.0 to 1.0)")
defaultFLAGS.define("gpuName", "/gpu:0", "GPU device name")
defaultFLAGS.define("lr", 1e-5, "learning rate")
defaultFLAGS.define("keep",20,"Number of most recent training results to save")
defaultFLAGS.define("batch", 16, "batch size")
defaultFLAGS.define("epoch", 1000, "number of epoch")
defaultFLAGS.define("save", 2000, "save checkpoint every ? training examples")
defaultFLAGS.define("demo", '', "demo on webcam")
defaultFLAGS.define("profile", False, "profile")
defaultFLAGS.define("json", False, "Outputs bounding box information in json format.")
defaultFLAGS.define("saveVideo", False, "Records video from input video or camera")
defaultFLAGS.define("pbLoad", "", "path to .pb protobuf file (metaLoad must also be specified)")
defaultFLAGS.define("metaLoad", "", "path to .meta file generated during --savepb that corresponds to .pb file")
