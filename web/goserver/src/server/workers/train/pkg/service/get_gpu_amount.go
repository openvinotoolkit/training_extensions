package service

import (
	"context"
	"fmt"

	"github.com/jaypipes/ghw"
)

type GetGpuAmountRequestData struct {
}

type GetGpuAmountResponseData struct {
	Amount int `json:"amount"`
}

func (s *basicTrainModelService) GetGpuAmount(ctx context.Context, req GetGpuAmountRequestData) (interface{}, error) {
	gpu, err := ghw.GPU()
	if err != nil {
		fmt.Printf("Error getting GPU info: %v", err)
	}
	return GetGpuAmountResponseData{
		Amount: len(gpu.GraphicsCards),
	}, nil
}
