package endpoint

import (
	"context"

	kitendpoint "server/kit/endpoint"
	"server/workers/train/pkg/service"
)

type Endpoints struct {
	GetGpuAmount kitendpoint.Endpoint
	RunCommands  kitendpoint.Endpoint
}

func New(s service.TrainModelService) Endpoints {
	eps := Endpoints{
		GetGpuAmount: MakeGetGpuAmountEndpoint(s),
		RunCommands:  MakeRunCommandsEndpoint(s),
	}

	return eps
}

func MakeGetGpuAmountEndpoint(s service.TrainModelService) kitendpoint.Endpoint {
	return func(ctx context.Context, req interface{}) chan kitendpoint.Response {
		returnChan := make(chan kitendpoint.Response)
		go func() {
			defer close(returnChan)
			resp, err := s.GetGpuAmount(ctx, req.(service.GetGpuAmountRequestData))
			if resp == nil {
				resp = service.GetGpuAmountResponseData{}
			}
			returnChan <- kitendpoint.Response{
				Data:   resp.(service.GetGpuAmountResponseData),
				Err:    err,
				IsLast: true,
			}
		}()
		return returnChan

	}
}

func MakeRunCommandsEndpoint(s service.TrainModelService) kitendpoint.Endpoint {
	return func(ctx context.Context, req interface{}) chan kitendpoint.Response {
		returnChan := make(chan kitendpoint.Response)
		go func() {
			defer close(returnChan)
			resp, err := s.RunCommands(ctx, req.(service.RunCommandsRequestData))
			returnChan <- kitendpoint.Response{
				Data:   resp,
				Err:    err,
				IsLast: true,
			}
		}()
		return returnChan

	}
}
