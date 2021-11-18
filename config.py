import os

cur_path = os.path.dirname(os.path.realpath(__file__))

sample_meta_path = os.path.join(cur_path, 'data/sample/meta.json')

black_sample_path = os.path.join(cur_path, 'data/sample/black')
black_output_path = os.path.join(cur_path, 'data/output/black')

white_sample_path = os.path.join(cur_path, 'data/sample/white')
white_output_path = os.path.join(cur_path, 'data/output/white')

black_ast_path = os.path.join(cur_path, 'data/ast/black')
white_ast_path = os.path.join(cur_path, 'data/ast/white')

tfidf_file = os.path.join(cur_path, 'data/pkl/tfidf.pkl')
embed_file = os.path.join(cur_path, 'data/pkl/embed.pkl')

ast2vec_logdir = os.path.join(cur_path, 'logs/ast2vec')

tbcnn_log_dir = os.path.join(cur_path, 'logs/tbcnn')
tbcnn_caputo_log_dir = os.path.join(cur_path, 'logs/tbcnn_caputo')

tbcnn_ckpt_filename = os.path.join(cur_path, 'data/model/tbcnn/tbcnn.ckpt')
tbcnn_caputo_ckpt_filename = os.path.join(cur_path, 'data/model/tbcnn_caputo/tbcnn_caputo.ckpt')