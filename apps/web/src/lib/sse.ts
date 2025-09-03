// Real-time KOL discovery updates (PoC1)
export class KOLDiscoveryStream {
  private eventSource: EventSource | null = null
  
  connect(onKOLFound: (kol: any) => void, onError?: (error: Event) => void) {
    this.eventSource = new EventSource(`${process.env.NEXT_PUBLIC_ML_SERVICE_URL}/discovery/stream`)
    
    this.eventSource.onmessage = (event) => {
      const newKOL = JSON.parse(event.data)
      onKOLFound(newKOL)
    }
    
    this.eventSource.onerror = (error) => {
      console.error('KOL discovery stream error:', error)
      onError?.(error)
    }
  }
  
  disconnect() {
    this.eventSource?.close()
    this.eventSource = null
  }
}

export const kolDiscoveryStream = new KOLDiscoveryStream()