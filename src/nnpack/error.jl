struct NNPACKError <: Exception
    code::nnp_status
    msg::AbstractString
end

Base.show(io::IO, err::NNPACKError) = print(io, "NNPACKError(code $(err.code), $(err.msg))")

function NNPACKError(status::nnp_status)
    msg = "NNPACK STATUS SUCCESS"
    if status == nnp_status_invalid_batch_size
        msg = "NNPACK STATUS INVALID BATCH SIZE"
    elseif status == nnp_status_invalid_channels
        msg = "NNPACK STATUS INVALID CHANNELS"
    elseif status == nnp_status_invalid_input_channels
        msg = "NNPACK STATUS INVALID INPUT CHANNELS"
    elseif status == nnp_status_invalid_output_channels
        msg = "NNPACK STATUS INVALID OUTPUT CHANNELS"
    elseif status == nnp_status_invalid_input_size
	    msg = "NNPACK STATUS INVALID INPUT SIZE"
    elseif status == nnp_status_invalid_input_stride
        msg = "NNPACK STATUS INVALID INPUT STRIDE"
    elseif status == nnp_status_invalid_input_padding
        msg = "NNPACK STATUS INVALID INPUT PADDING"
    elseif status == nnp_status_invalid_kernel_size
        msg = "NNPACK STATUS INVALID KERNEL SIZE"
    elseif status == nnp_status_invalid_pooling_size
        msg = "NNPACK STATUS INVALID POOLING SIZE"
    elseif status == nnp_status_invalid_pooling_stride
        msg = "NNPACK STATUS INVALID POOLING STRIDE"
    elseif status == nnp_status_invalid_algorithm
        msg = "NNPACK STATUS INVALID ALGORITHM"
    elseif status == nnp_status_invalid_transform_strategy
        msg = "NNPACK STATUS INVALID TRANSFORM STRATEGY"
    elseif status == nnp_status_invalid_output_subsampling
        msg = "NNPACK STATUS INVALID OUTPUT SUBSAMPLING"
    elseif status == nnp_status_invalid_activation
        msg = "NNPACK STATUS INVALID ACTIVATION"
    elseif status == nnp_status_invalid_activation_parameters
        msg = "NNPACK STATUS INVALID ACTIVATION PARAMETERS"
    elseif status == nnp_status_unsupported_input_size
        msg = "NNPACK STATUS UNSUPPORTED INPUT SIZE"
    elseif status == nnp_status_unsupported_input_stride
        msg = "NNPACK STATUS UNSUPPORTED INPUT STRIDE"
    elseif status == nnp_status_unsupported_input_padding
        msg = "NNPACK STATUS UNSUPPORTED INPUT PADDING"
    elseif status == nnp_status_unsupported_kernel_size
        msg = "NNPACK STATUS UNSUPPORTED KERNEL SIZE"
    elseif status == nnp_status_unsupported_pooling_size
        msg = "NNPACK STATUS UNSUPPORTED POOLING SIZE"
    elseif status == nnp_status_unsupported_pooling_stride
        msg = "NNPACK STATUS UNSUPPORTED POOLING STRIDE"
    elseif status == nnp_status_unsupported_algorithm
        msg = "NNPACK STATUS UNSUPPORTED ALGORITHM"
    elseif status == nnp_status_unsupported_transform_strategy
        msg = "NNPACK STATUS UNSUPPORTED TRANSFORM STRATEGY"
    elseif status == nnp_status_unsupported_activation
        msg = "NNPACK STATUS UNSUPPORTED ACTIVATION"
    elseif status == nnp_status_unsupported_activation_parameters
        msg = "NNPACK STATUS UNSUPPORTED ACTIVATION PARAMETERS"
    elseif status == nnp_status_uninitialized
        msg = "NNPACK STATUS UNINITIALIZED"
    elseif status == nnp_status_unsupported_hardware
        msg = "NNPACK STATUS UNSUPPORTED HARDWARE"
    elseif status == nnp_status_out_of_memory
        msg = "NNPACK STATUS OUT OF MEMORY"
    elseif status == nnp_status_insufficient_buffer
        msg = "NNPACK STATUS INSUFFICIENT BUFFER"
    elseif status == nnp_status_misaligned_buffer
        msg = "NNPACK STATUS MISALIGNED BUFFER"
    end
    NNPACKError(status, msg)
end

macro check(nnp_func)
    quote
        local err::nnp_status
        err = $(esc(nnp_func))
        if err != nnp_status_success
            throw(NNPACKError(err))
        end
        err
    end
end
