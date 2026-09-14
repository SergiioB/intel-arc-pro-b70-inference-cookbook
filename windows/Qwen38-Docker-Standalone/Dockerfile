FROM vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f

COPY patches/patch_mtp_nightly.py /opt/qwen38/patch_mtp_nightly.py
COPY patches/patch_mtp_boundary.py /opt/qwen38/patch_mtp_boundary.py
COPY patches/patch_gdn_mixed_split_v5.py /opt/qwen38/patch_gdn_mixed_split_v5.py
COPY patches/patch_draft_lmhead_int4.py /opt/qwen38/patch_draft_lmhead_int4.py
COPY patches/patch_draft_mtp_int4.py /opt/qwen38/patch_draft_mtp_int4.py
COPY container/start.sh /opt/qwen38/start.sh
COPY container/diagnose.py /opt/qwen38/diagnose.py
COPY container/patch_xpu_single_gpu_warmup.py /opt/qwen38/patch_xpu_single_gpu_warmup.py
COPY container/prepare_low_reasoning_template.py /opt/qwen38/prepare_low_reasoning_template.py

# Git for Windows may COPY CRLF scripts. Strip CR so start.sh has a Unix shebang.
RUN sed -i 's/\r$//' /opt/qwen38/start.sh /opt/qwen38/*.py \
    && chmod 0755 /opt/qwen38/start.sh \
    && python -m py_compile /opt/qwen38/patch_mtp_nightly.py \
       /opt/qwen38/patch_mtp_boundary.py /opt/qwen38/patch_gdn_mixed_split_v5.py \
       /opt/qwen38/patch_draft_lmhead_int4.py /opt/qwen38/patch_draft_mtp_int4.py \
       /opt/qwen38/diagnose.py \
       /opt/qwen38/patch_xpu_single_gpu_warmup.py \
       /opt/qwen38/prepare_low_reasoning_template.py

EXPOSE 8000
ENTRYPOINT ["/opt/qwen38/start.sh"]
