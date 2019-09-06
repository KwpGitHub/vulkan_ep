
CALL cd .\onnx_builder
CALL python .\onnx_builder.py
CALL cd ../_backend
CALL python .\compile_shader.py
CALL cd ..