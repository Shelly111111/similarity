import paddle
import paddle.fluid as fluid
import work.similarity_net.config as config
import work.similarity_net.utils as utils
import work.models.matching.paddle_layers as layers
import work.similarity_net.run_classifier as run
import jieba

from work.similarity_net.utils import ArgConfig


if __name__ == "__main__":

    args = ArgConfig()
    args = args.build_conf()
    query = '后牌照怎么装'
    query=' '.join(jieba.cut(query,cut_all=True))
    try:
        if fluid.is_compiled_with_cuda() != True and args.use_cuda == True:
            print(
                "\nYou can not set use_cuda = True in the model because you are using paddlepaddle-cpu.\nPlease: 1. Install paddlepaddle-gpu to run your models on GPU or 2. Set use_cuda = False to run models on CPU.\n"
            )

            sys.exit(1)
    except Exception as e:
        pass
    utils.init_log("./log/TextSimilarityNet")
    conf_dict = config.SimNetConfig(args)

    run.infer(args,query)